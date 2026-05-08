#include "darknet_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static inline int yolov9_channel_index(const Darknet::Layer & from, const int b, const int c, const int y, const int x)
	{
		return b * from.outputs + c * from.out_h * from.out_w + y * from.out_w + x;
	}

	static inline float sigmoid(const float x)
	{
		return logistic_activate(x);
	}

	static inline int selected_branch_offset(const Darknet::Layer & l)
	{
		return l.inference_branch * l.n;
	}

	static inline void softmax_bins(const float * logits, const int reg_max, float * probs)
	{
		if (reg_max <= 0)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d must be positive", reg_max);
		}

		float max_value = logits[0];
		for (int i = 1; i < reg_max; ++i)
		{
			max_value = std::max(max_value, logits[i]);
		}

		float sum = 0.0f;
		for (int i = 0; i < reg_max; ++i)
		{
			probs[i] = std::exp(logits[i] - max_value);
			sum += probs[i];
		}

		const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
		for (int i = 0; i < reg_max; ++i)
		{
			probs[i] *= inv_sum;
		}
	}

	static inline void get_dfl_distances(const Darknet::Layer & from, const int b, const int x, const int y, const int reg_max, float distances[4])
	{
		float logits[64];
		float probs[64];
		if (reg_max <= 0 or reg_max > static_cast<int>(sizeof(logits) / sizeof(logits[0])))
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d exceeds local DFL buffer", reg_max);
		}

		for (int side = 0; side < 4; ++side)
		{
			for (int bin = 0; bin < reg_max; ++bin)
			{
				const int channel = side * reg_max + bin;
				logits[bin] = from.output[yolov9_channel_index(from, b, channel, y, x)];
			}
			softmax_bins(logits, reg_max, probs);

			float distance = 0.0f;
			for (int bin = 0; bin < reg_max; ++bin)
			{
				distance += static_cast<float>(bin) * probs[bin];
			}
			distances[side] = distance;
		}
	}

	static inline Darknet::Box decode_prediction(const Darknet::Layer & from, const int b, const int x, const int y, const int stride, const int reg_max, const int netw, const int neth)
	{
		float distances[4];
		get_dfl_distances(from, b, x, y, reg_max, distances);
		return yolov9_dist2bbox(x + 0.5f, y + 0.5f, distances, static_cast<float>(stride), netw, neth);
	}

	static inline int best_class_score(const Darknet::Layer & from, const int b, const int x, const int y, const int classes, const int reg_max, float & best_score)
	{
		best_score = 0.0f;
		int best_class = -1;
		for (int class_id = 0; class_id < classes; ++class_id)
		{
			const int channel = 4 * reg_max + class_id;
			const float score = sigmoid(from.output[yolov9_channel_index(from, b, channel, y, x)]);
			if (score > best_score)
			{
				best_score = score;
				best_class = class_id;
			}
		}
		return best_class;
	}

	static inline void add_class_delta(Darknet::Layer & from, const int b, const int x, const int y, const int class_id, const int classes, const int reg_max, const float target, const float scale)
	{
		if (class_id < 0 or class_id >= classes)
		{
			return;
		}

		const int channel = 4 * reg_max + class_id;
		const int index = yolov9_channel_index(from, b, channel, y, x);
		const float prediction = sigmoid(from.output[index]);
		from.delta[index] += scale * (target - prediction);
	}

	static inline void add_dfl_delta(Darknet::Layer & from, const int b, const int x, const int y, const int side, const int reg_max, const float target_distance, const float scale)
	{
		float logits[64];
		float probs[64];
		if (reg_max <= 0 or reg_max > static_cast<int>(sizeof(logits) / sizeof(logits[0])))
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d exceeds local DFL buffer", reg_max);
		}

		for (int bin = 0; bin < reg_max; ++bin)
		{
			const int channel = side * reg_max + bin;
			logits[bin] = from.output[yolov9_channel_index(from, b, channel, y, x)];
		}
		softmax_bins(logits, reg_max, probs);

		const float clipped = std::max(0.0f, std::min(target_distance, static_cast<float>(reg_max) - 1.01f));
		const int left_bin = static_cast<int>(std::floor(clipped));
		const int right_bin = std::min(left_bin + 1, reg_max - 1);
		const float right_weight = clipped - left_bin;
		const float left_weight = 1.0f - right_weight;

		for (int bin = 0; bin < reg_max; ++bin)
		{
			float target = 0.0f;
			if (bin == left_bin)
			{
				target += left_weight;
			}
			if (bin == right_bin)
			{
				target += right_weight;
			}

			const int channel = side * reg_max + bin;
			const int index = yolov9_channel_index(from, b, channel, y, x);
			from.delta[index] += scale * (target - probs[bin]);
		}
	}

	static void zero_input_deltas(const Darknet::Layer & l, Darknet::NetworkState state)
	{
		for (int i = 0; i < l.total; ++i)
		{
			Darknet::Layer & from = state.net.layers[l.input_layers[i]];
			if (from.delta)
			{
				scal_cpu(from.outputs * from.batch, 0.0f, from.delta, 1);
			}
		}
	}

	static void train_yolov9_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		if (state.truth == nullptr)
		{
			return;
		}

		zero_input_deltas(l, state);

		float cls_loss = 0.0f;
		float dfl_loss = 0.0f;
		float box_loss = 0.0f;
		int assignments = 0;

		const float negative_scale = l.cls_normalizer / std::max(1, l.classes * 16);
		const int branch_offset = selected_branch_offset(l);
		for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
		{
			Darknet::Layer & from = state.net.layers[l.input_layers[branch_offset + scale_idx]];
			for (int b = 0; b < l.batch; ++b)
			{
				for (int y = 0; y < from.out_h; ++y)
				{
					for (int x = 0; x < from.out_w; ++x)
					{
						for (int class_id = 0; class_id < l.classes; ++class_id)
						{
							const int channel = 4 * l.reg_max + class_id;
							const int index = yolov9_channel_index(from, b, channel, y, x);
							const float prediction = sigmoid(from.output[index]);
							from.delta[index] += negative_scale * (0.0f - prediction);
							cls_loss += -std::log(std::max(1.0f - prediction, 1e-6f));
						}
					}
				}
			}
		}

		for (int b = 0; b < l.batch; ++b)
		{
			for (int t = 0; t < l.max_boxes; ++t)
			{
				const float * truth_ptr = state.truth + b * l.truths + t * l.truth_size;
				const Darknet::Box truth = float_to_box_stride(truth_ptr, 1);
				if (truth.x <= 0.0f)
				{
					break;
				}

				const int class_id = static_cast<int>(truth_ptr[4]);
				if (class_id < 0 or class_id >= l.classes)
				{
					darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d", class_id);
				}

				const float truth_w_pixels = truth.w * state.net.w;
				const float truth_h_pixels = truth.h * state.net.h;
				const float truth_size_pixels = std::max(truth_w_pixels, truth_h_pixels);

				int best_scale = 0;
				float best_scale_distance = FLT_MAX;
				for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
				{
					const float scale_distance = std::fabs(std::log2(std::max(1.0f, truth_size_pixels) / std::max(1, l.strides[scale_idx])));
					if (scale_distance < best_scale_distance)
					{
						best_scale_distance = scale_distance;
						best_scale = scale_idx;
					}
				}

				Darknet::Layer & from = state.net.layers[l.input_layers[branch_offset + best_scale]];
				const int stride = std::max(1, l.strides[best_scale]);
				const int cell_x = std::max(0, std::min(from.out_w - 1, static_cast<int>(truth.x * state.net.w / stride)));
				const int cell_y = std::max(0, std::min(from.out_h - 1, static_cast<int>(truth.y * state.net.h / stride)));

				const float anchor_x = cell_x + 0.5f;
				const float anchor_y = cell_y + 0.5f;
				const float x1 = (truth.x - truth.w * 0.5f) * state.net.w / stride;
				const float y1 = (truth.y - truth.h * 0.5f) * state.net.h / stride;
				const float x2 = (truth.x + truth.w * 0.5f) * state.net.w / stride;
				const float y2 = (truth.y + truth.h * 0.5f) * state.net.h / stride;
				const float target_distances[4] =
				{
					anchor_x - x1,
					anchor_y - y1,
					x2 - anchor_x,
					y2 - anchor_y
				};

				for (int side = 0; side < 4; ++side)
				{
					add_dfl_delta(from, b, cell_x, cell_y, side, l.reg_max, target_distances[side], l.dfl_normalizer);
					dfl_loss += std::max(0.0f, target_distances[side]);
				}

				add_class_delta(from, b, cell_x, cell_y, class_id, l.classes, l.reg_max, 1.0f, l.cls_normalizer);
				float best_score = 0.0f;
				best_class_score(from, b, cell_x, cell_y, l.classes, l.reg_max, best_score);
				cls_loss += -std::log(std::max(best_score, 1e-6f));

				const Darknet::Box pred = decode_prediction(from, b, cell_x, cell_y, stride, l.reg_max, state.net.w, state.net.h);
				const float iou = box_iou(pred, truth);
				box_loss += 1.0f - iou;
				assignments++;
			}
		}

		const float normalizer = 1.0f / std::max(1, assignments);
		l.cost[0] = normalizer * (l.box_normalizer * box_loss + l.cls_normalizer * cls_loss + l.dfl_normalizer * dfl_loss);
	}
}


Darknet::Layer make_yolov9_layer(int batch, int classes, int reg_max, int branch_count, int inference_branch, int input_count, int *input_layers, int *input_sizes, int *strides, int max_boxes, int total_points)
{
	TAT(TATPARMS);

	const int scale_count = input_count / branch_count;

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::YOLOV9;
	l.batch = batch;
	l.n = scale_count;
	l.total = input_count;
	l.input_layers = input_layers;
	l.input_sizes = input_sizes;
	l.strides = strides;
	l.classes = classes;
	l.reg_max = reg_max;
	l.coords = 4;
	l.objectness = 0;
	l.max_boxes = max_boxes;
	l.truth_size = 4 + 2;
	l.truths = l.max_boxes * l.truth_size;
	l.branch_count = branch_count;
	l.inference_branch = inference_branch;
	l.aux_loss_weight = 0.25f;
	l.box_normalizer = 7.5f;
	l.cls_normalizer = 0.5f;
	l.dfl_normalizer = 1.5f;
	l.tal_topk = 10;
	l.tal_alpha = 0.5f;
	l.tal_beta = 6.0f;

	l.inputs = 0;
	for (int i = 0; i < input_count; ++i)
	{
		l.inputs += input_sizes[i];
	}

	l.outputs = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int i = 0; i < l.n; ++i)
	{
		l.outputs += input_sizes[branch_offset + i];
	}
	l.out_w = 1;
	l.out_h = 1;
	l.out_c = l.outputs;

	l.cost = (float*)xcalloc(1, sizeof(float));
	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.forward = forward_yolov9_layer;
	l.backward = backward_yolov9_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_yolov9_layer_gpu;
	l.backward_gpu = backward_yolov9_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
#endif

	*cfg_and_state.output
		<< "yolov9                       "
		<< "scales=" << scale_count
		<< ", branches=" << branch_count
		<< ", classes=" << classes
		<< ", reg_max=" << reg_max
		<< ", points=" << total_points
		<< ", outputs=" << l.outputs
		<< std::endl;

	return l;
}


void resize_yolov9_layer(Darknet::Layer * l, Darknet::Network * net)
{
	TAT(TATPARMS);

	l->inputs = 0;
	for (int i = 0; i < l->total; ++i)
	{
		const Darknet::Layer & from = net->layers[l->input_layers[i]];
		l->input_sizes[i] = from.outputs;
		l->inputs += from.outputs;
	}

	l->outputs = 0;
	const int branch_offset = selected_branch_offset(*l);
	for (int i = 0; i < l->n; ++i)
	{
		const Darknet::Layer & from = net->layers[l->input_layers[branch_offset + i]];
		l->outputs += from.outputs;
	}
	l->out_w = 1;
	l->out_h = 1;
	l->out_c = l->outputs;

	l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
	l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
#endif
}


void forward_yolov9_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	scal_cpu(l.outputs * l.batch, 0.0f, l.delta, 1);

	int offset = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int i = 0; i < l.n; ++i)
	{
		const int input_index = l.input_layers[branch_offset + i];
#ifdef DARKNET_USE_MPS
		mps_flush_deferred_output(&state.net.layers[input_index]);
#endif
		const Darknet::Layer & from = state.net.layers[input_index];
		for (int b = 0; b < l.batch; ++b)
		{
			copy_cpu(from.outputs, from.output + b * from.outputs, 1, l.output + b * l.outputs + offset, 1);
		}
		offset += from.outputs;
	}

	if (state.train and not l.onlyforward)
	{
		train_yolov9_layer(l, state);
	}
}


void backward_yolov9_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	int offset = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int i = 0; i < l.n; ++i)
	{
		Darknet::Layer & from = state.net.layers[l.input_layers[branch_offset + i]];
		for (int b = 0; b < l.batch; ++b)
		{
			axpy_cpu(from.outputs, state.net.loss_scale, l.delta + b * l.outputs + offset, 1, from.delta + b * from.outputs, 1);
		}
		offset += from.outputs;
	}
}


float yolov9_dfl_project(const float * logits, int reg_max)
{
	float probs[64];
	if (reg_max <= 0 or reg_max > static_cast<int>(sizeof(probs) / sizeof(probs[0])))
	{
		darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d exceeds local DFL buffer", reg_max);
	}
	softmax_bins(logits, reg_max, probs);

	float value = 0.0f;
	for (int i = 0; i < reg_max; ++i)
	{
		value += static_cast<float>(i) * probs[i];
	}
	return value;
}


Darknet::Box yolov9_dist2bbox(float anchor_x, float anchor_y, const float distances[4], float stride, int netw, int neth)
{
	Darknet::Box b;
	const float x1 = (anchor_x - distances[0]) * stride;
	const float y1 = (anchor_y - distances[1]) * stride;
	const float x2 = (anchor_x + distances[2]) * stride;
	const float y2 = (anchor_y + distances[3]) * stride;

	b.x = ((x1 + x2) * 0.5f) / netw;
	b.y = ((y1 + y2) * 0.5f) / neth;
	b.w = (x2 - x1) / netw;
	b.h = (y2 - y1) / neth;
	return b;
}


int yolov9_num_detections(const Darknet::Network * net, const Darknet::Layer & l, float thresh)
{
	TAT(TATPARMS);

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const Darknet::Layer & from = net->layers[l.input_layers[branch_offset + scale_idx]];
		for (int y = 0; y < from.out_h; ++y)
		{
			for (int x = 0; x < from.out_w; ++x)
			{
				float best_score = 0.0f;
				best_class_score(from, 0, x, y, l.classes, l.reg_max, best_score);
				if (best_score > thresh)
				{
					count++;
				}
			}
		}
	}
	return count;
}


int yolov9_num_detections_v3(Darknet::Network * net, const int index, const float thresh, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	const Darknet::Layer & l = net->layers[index];
	int count = 0;
	int point_index = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const Darknet::Layer & from = net->layers[l.input_layers[branch_offset + scale_idx]];
		for (int y = 0; y < from.out_h; ++y)
		{
			for (int x = 0; x < from.out_w; ++x)
			{
				float best_score = 0.0f;
				best_class_score(from, 0, x, y, l.classes, l.reg_max, best_score);
				if (best_score > thresh)
				{
					Darknet::Output_Object oo;
					oo.layer_index = index;
					oo.n = scale_idx;
					oo.i = point_index;
					oo.obj_index = 0;
					cache.push_back(oo);
					count++;
				}
				point_index++;
			}
		}
	}
	return count;
}


int yolov9_num_detections_batch(const Darknet::Network * net, const Darknet::Layer & l, float thresh, int batch)
{
	TAT(TATPARMS);

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const Darknet::Layer & from = net->layers[l.input_layers[branch_offset + scale_idx]];
		for (int y = 0; y < from.out_h; ++y)
		{
			for (int x = 0; x < from.out_w; ++x)
			{
				float best_score = 0.0f;
				best_class_score(from, batch, x, y, l.classes, l.reg_max, best_score);
				if (best_score > thresh)
				{
					count++;
				}
			}
		}
	}
	return count;
}


int get_yolov9_detections(const Darknet::Network * net, const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter)
{
	TAT(TATPARMS);

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const Darknet::Layer & from = net->layers[l.input_layers[branch_offset + scale_idx]];
		const int stride = std::max(1, l.strides[scale_idx]);
		for (int y = 0; y < from.out_h; ++y)
		{
			for (int x = 0; x < from.out_w; ++x)
			{
				float best_score = 0.0f;
				const int best_class = best_class_score(from, 0, x, y, l.classes, l.reg_max, best_score);
				if (best_score <= thresh)
				{
					continue;
				}

				dets[count].bbox = decode_prediction(from, 0, x, y, stride, l.reg_max, netw, neth);
				dets[count].objectness = 1.0f;
				dets[count].classes = l.classes;
				dets[count].best_class_idx = best_class;

				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const int mapped_class_id = map ? map[class_id] : class_id;
					if (mapped_class_id < 0 or mapped_class_id >= l.classes)
					{
						continue;
					}
					const int channel = 4 * l.reg_max + class_id;
					const float score = sigmoid(from.output[yolov9_channel_index(from, 0, channel, y, x)]);
					dets[count].prob[mapped_class_id] = (score > thresh) ? score : 0.0f;
				}
				count++;
			}
		}
	}

	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
	return count;
}


int get_yolov9_detections_batch(const Darknet::Network * net, const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter, int batch)
{
	TAT(TATPARMS);

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const Darknet::Layer & from = net->layers[l.input_layers[branch_offset + scale_idx]];
		const int stride = std::max(1, l.strides[scale_idx]);
		for (int y = 0; y < from.out_h; ++y)
		{
			for (int x = 0; x < from.out_w; ++x)
			{
				float best_score = 0.0f;
				const int best_class = best_class_score(from, batch, x, y, l.classes, l.reg_max, best_score);
				if (best_score <= thresh)
				{
					continue;
				}

				dets[count].bbox = decode_prediction(from, batch, x, y, stride, l.reg_max, netw, neth);
				dets[count].objectness = 1.0f;
				dets[count].classes = l.classes;
				dets[count].best_class_idx = best_class;

				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const int mapped_class_id = map ? map[class_id] : class_id;
					if (mapped_class_id < 0 or mapped_class_id >= l.classes)
					{
						continue;
					}
					const int channel = 4 * l.reg_max + class_id;
					const float score = sigmoid(from.output[yolov9_channel_index(from, batch, channel, y, x)]);
					dets[count].prob[mapped_class_id] = (score > thresh) ? score : 0.0f;
				}
				count++;
			}
		}
	}

	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
	return count;
}


#ifdef DARKNET_GPU
void forward_yolov9_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int branch_offset = selected_branch_offset(l);
	for (int i = 0; i < l.n; ++i)
	{
		Darknet::Layer & from = state.net.layers[l.input_layers[branch_offset + i]];
		cuda_pull_array(from.output_gpu, from.output, from.batch * from.outputs);
	}

	float * truth_cpu = nullptr;
	Darknet::NetworkState cpu_state = state;
	if (state.truth)
	{
		const int num_truth = l.batch * l.truths;
		truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
		cpu_state.truth = truth_cpu;
	}
	cpu_state.input = l.output;

	forward_yolov9_layer(l, cpu_state);

	cuda_push_array(l.output_gpu, l.output, l.batch * l.outputs);
	if (state.train and not l.onlyforward)
	{
		cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
		for (int i = 0; i < l.n; ++i)
		{
			Darknet::Layer & from = state.net.layers[l.input_layers[branch_offset + i]];
			cuda_push_array(from.delta_gpu, from.delta, from.batch * from.outputs);
		}
	}

	if (truth_cpu)
	{
		free(truth_cpu);
	}
}


void backward_yolov9_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	int offset = 0;
	const int branch_offset = selected_branch_offset(l);
	for (int i = 0; i < l.n; ++i)
	{
		Darknet::Layer & from = state.net.layers[l.input_layers[branch_offset + i]];
		for (int b = 0; b < l.batch; ++b)
		{
			axpy_ongpu(from.outputs, state.net.loss_scale, l.delta_gpu + b * l.outputs + offset, 1, from.delta_gpu + b * from.outputs, 1);
		}
		offset += from.outputs;
	}
}
#endif
