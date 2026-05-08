#include "darknet_internal.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

float yolov9_dfl_cross_entropy_delta(const float *logits, int reg_max, float target_distance, float scale, float *delta);

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

	struct BranchPoint
	{
		int input_index = -1;
		int scale_idx = 0;
		int x = 0;
		int y = 0;
		int stride = 1;
		float anchor_x = 0.0f;
		float anchor_y = 0.0f;
		Darknet::Box decoded_box = {};
	};

	struct GroundTruth
	{
		int class_id = -1;
		Darknet::Box box = {};
		float x1 = 0.0f;
		float y1 = 0.0f;
		float x2 = 0.0f;
		float y2 = 0.0f;
	};

	struct AssignmentCandidate
	{
		int truth_idx = -1;
		int point_idx = -1;
		float metric = 0.0f;
		float overlap = 0.0f;
	};

	struct Assignment
	{
		int truth_idx = -1;
		float metric = 0.0f;
		float overlap = 0.0f;
		float target_score = 0.0f;
	};

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

	static inline float class_logit(const Darknet::Layer & from, const int b, const int x, const int y, const int class_id, const int reg_max)
	{
		const int channel = 4 * reg_max + class_id;
		return from.output[yolov9_channel_index(from, b, channel, y, x)];
	}

	static inline float class_score(const Darknet::Layer & from, const int b, const int x, const int y, const int class_id, const int reg_max)
	{
		return sigmoid(class_logit(from, b, x, y, class_id, reg_max));
	}

	static inline float bce_with_logits(const float logit, const float target)
	{
		const float max_value = std::max(logit, 0.0f);
		return max_value - logit * target + std::log1p(std::exp(-std::fabs(logit)));
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

	static inline float add_dfl_delta(Darknet::Layer & from, const int b, const int x, const int y, const int side, const int reg_max, const float target_distance, const float scale)
	{
		float logits[64];
		float deltas[64];
		if (reg_max <= 0 or reg_max > static_cast<int>(sizeof(logits) / sizeof(logits[0])))
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d exceeds local DFL buffer", reg_max);
		}

		for (int bin = 0; bin < reg_max; ++bin)
		{
			const int channel = side * reg_max + bin;
			logits[bin] = from.output[yolov9_channel_index(from, b, channel, y, x)];
		}
		const float loss = yolov9_dfl_cross_entropy_delta(logits, reg_max, target_distance, scale, deltas);

		for (int bin = 0; bin < reg_max; ++bin)
		{
			const int channel = side * reg_max + bin;
			const int index = yolov9_channel_index(from, b, channel, y, x);
			from.delta[index] += deltas[bin];
		}
		return loss;
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

	static std::vector<GroundTruth> load_truths_for_batch(const Darknet::Layer & l, const Darknet::NetworkState & state, const int b)
	{
		std::vector<GroundTruth> truths;
		for (int t = 0; t < l.max_boxes; ++t)
		{
			const float * truth_ptr = state.truth + b * l.truths + t * l.truth_size;
			const Darknet::Box box = float_to_box_stride(truth_ptr, 1);
			if (box.x <= 0.0f)
			{
				break;
			}

			const int class_id = static_cast<int>(truth_ptr[4]);
			if (class_id < 0 or class_id >= l.classes)
			{
				darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d", class_id);
			}

			GroundTruth truth;
			truth.class_id = class_id;
			truth.box = box;
			truth.x1 = (box.x - box.w * 0.5f) * state.net.w;
			truth.y1 = (box.y - box.h * 0.5f) * state.net.h;
			truth.x2 = (box.x + box.w * 0.5f) * state.net.w;
			truth.y2 = (box.y + box.h * 0.5f) * state.net.h;
			truths.push_back(truth);
		}
		return truths;
	}

	static std::vector<BranchPoint> collect_branch_points(const Darknet::Layer & l, Darknet::NetworkState state, const int branch, const int b)
	{
		std::vector<BranchPoint> points;
		for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
		{
			const int input_slot = branch * l.n + scale_idx;
			const int input_index = l.input_layers[input_slot];
			const Darknet::Layer & from = state.net.layers[input_index];
			points.reserve(points.size() + static_cast<std::size_t>(from.out_w) * static_cast<std::size_t>(from.out_h));
			const int stride = std::max(1, l.strides[scale_idx]);
			for (int y = 0; y < from.out_h; ++y)
			{
				for (int x = 0; x < from.out_w; ++x)
				{
					BranchPoint point;
					point.input_index = input_index;
					point.scale_idx = scale_idx;
					point.x = x;
					point.y = y;
					point.stride = stride;
					point.anchor_x = x + 0.5f;
					point.anchor_y = y + 0.5f;
					point.decoded_box = decode_prediction(from, b, x, y, stride, l.reg_max, state.net.w, state.net.h);
					points.push_back(point);
				}
			}
		}
		return points;
	}

	static inline bool anchor_inside_truth(const BranchPoint & point, const GroundTruth & truth)
	{
		const float anchor_x = point.anchor_x * point.stride;
		const float anchor_y = point.anchor_y * point.stride;
		return anchor_x > truth.x1 and anchor_y > truth.y1 and anchor_x < truth.x2 and anchor_y < truth.y2;
	}

	static std::vector<Assignment> assign_branch_targets(
		const Darknet::Layer & l,
		Darknet::NetworkState state,
		const std::vector<BranchPoint> & points,
		const std::vector<GroundTruth> & truths,
		const int b)
	{
		std::vector<Assignment> assignments(points.size());
		if (points.empty() or truths.empty())
		{
			return assignments;
		}

		std::vector<std::vector<AssignmentCandidate>> topk_per_truth(truths.size());
		const int topk = std::max(1, l.tal_topk);
		for (std::size_t truth_idx = 0; truth_idx < truths.size(); ++truth_idx)
		{
			std::vector<AssignmentCandidate> candidates;
			const GroundTruth & truth = truths[truth_idx];
			for (std::size_t point_idx = 0; point_idx < points.size(); ++point_idx)
			{
				const BranchPoint & point = points[point_idx];
				if (not anchor_inside_truth(point, truth))
				{
					continue;
				}

				const Darknet::Layer & from = state.net.layers[point.input_index];
				const float score = class_score(from, b, point.x, point.y, truth.class_id, l.reg_max);
				const float overlap = std::max(0.0f, box_iou_kind(point.decoded_box, truth.box, CIOU));
				if (overlap <= 0.0f or score <= 0.0f)
				{
					continue;
				}

				const float metric = std::pow(score, l.tal_alpha) * std::pow(overlap, l.tal_beta);
				if (metric <= 1.0e-12f or not std::isfinite(metric))
				{
					continue;
				}
				candidates.push_back(AssignmentCandidate{static_cast<int>(truth_idx), static_cast<int>(point_idx), metric, overlap});
			}

			if (candidates.empty())
			{
				continue;
			}
			std::sort(
				candidates.begin(),
				candidates.end(),
				[](const AssignmentCandidate & lhs, const AssignmentCandidate & rhs)
				{
					return lhs.metric > rhs.metric;
				});
			if (static_cast<int>(candidates.size()) > topk)
			{
				candidates.resize(topk);
			}
			topk_per_truth[truth_idx] = candidates;
		}

		for (const auto & candidates : topk_per_truth)
		{
			for (const AssignmentCandidate & candidate : candidates)
			{
				Assignment & current = assignments[candidate.point_idx];
				if (current.truth_idx < 0 or candidate.overlap > current.overlap)
				{
					current.truth_idx = candidate.truth_idx;
					current.metric = candidate.metric;
					current.overlap = candidate.overlap;
				}
			}
		}

		std::vector<float> max_metric_by_truth(truths.size(), 0.0f);
		std::vector<float> max_overlap_by_truth(truths.size(), 0.0f);
		for (const Assignment & assignment : assignments)
		{
			if (assignment.truth_idx < 0)
			{
				continue;
			}
			max_metric_by_truth[assignment.truth_idx] = std::max(max_metric_by_truth[assignment.truth_idx], assignment.metric);
			max_overlap_by_truth[assignment.truth_idx] = std::max(max_overlap_by_truth[assignment.truth_idx], assignment.overlap);
		}

		for (Assignment & assignment : assignments)
		{
			if (assignment.truth_idx < 0)
			{
				continue;
			}
			const float max_metric = max_metric_by_truth[assignment.truth_idx];
			const float max_overlap = max_overlap_by_truth[assignment.truth_idx];
			if (max_metric <= 0.0f)
			{
				assignment.truth_idx = -1;
				assignment.target_score = 0.0f;
				continue;
			}
			assignment.target_score = std::min(1.0f, assignment.metric * max_overlap / (max_metric + 1.0e-9f));
		}

		return assignments;
	}

	static float box_loss_from_distances(
		const BranchPoint & point,
		const float distances[4],
		const GroundTruth & truth,
		const IOU_LOSS iou_loss,
		const int netw,
		const int neth)
	{
		const Darknet::Box pred = yolov9_dist2bbox(point.anchor_x, point.anchor_y, distances, static_cast<float>(point.stride), netw, neth);
		const float overlap = box_iou_kind(pred, truth.box, iou_loss);
		return 1.0f - overlap;
	}

	static float add_ciou_projection_delta(
		Darknet::Layer & from,
		const Darknet::Layer & l,
		const BranchPoint & point,
		const int b,
		const GroundTruth & truth,
		const float scale,
		const int netw,
		const int neth)
	{
		float distances[4];
		get_dfl_distances(from, b, point.x, point.y, l.reg_max, distances);
		const float base_loss = box_loss_from_distances(point, distances, truth, l.iou_loss, netw, neth);
		const float eps = 0.05f;

		for (int side = 0; side < 4; ++side)
		{
			float side_logits[64];
			float side_probs[64];
			if (l.reg_max <= 0 or l.reg_max > static_cast<int>(sizeof(side_logits) / sizeof(side_logits[0])))
			{
				darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d exceeds local DFL buffer", l.reg_max);
			}
			for (int bin = 0; bin < l.reg_max; ++bin)
			{
				const int channel = side * l.reg_max + bin;
				side_logits[bin] = from.output[yolov9_channel_index(from, b, channel, point.y, point.x)];
			}
			softmax_bins(side_logits, l.reg_max, side_probs);

			float plus_distances[4] = {distances[0], distances[1], distances[2], distances[3]};
			float minus_distances[4] = {distances[0], distances[1], distances[2], distances[3]};
			plus_distances[side] = std::min(static_cast<float>(l.reg_max) - 1.0f, distances[side] + eps);
			minus_distances[side] = std::max(0.0f, distances[side] - eps);
			const float actual_eps = plus_distances[side] - minus_distances[side];
			if (actual_eps <= 0.0f)
			{
				continue;
			}

			const float loss_plus = box_loss_from_distances(point, plus_distances, truth, l.iou_loss, netw, neth);
			const float loss_minus = box_loss_from_distances(point, minus_distances, truth, l.iou_loss, netw, neth);
			const float dloss_ddistance = (loss_plus - loss_minus) / actual_eps;

			for (int bin = 0; bin < l.reg_max; ++bin)
			{
				const float dproject_dlogit = side_probs[bin] * (static_cast<float>(bin) - distances[side]);
				const int channel = side * l.reg_max + bin;
				const int index = yolov9_channel_index(from, b, channel, point.y, point.x);
				from.delta[index] += -scale * dloss_ddistance * dproject_dlogit;
			}
		}

		return scale * base_loss;
	}

	static float train_yolov9_branch(Darknet::Layer & l, Darknet::NetworkState state, const int branch)
	{
		const float branch_weight = (branch == l.inference_branch) ? 1.0f : l.aux_loss_weight;
		float branch_loss = 0.0f;

		for (int b = 0; b < l.batch; ++b)
		{
			const std::vector<GroundTruth> truths = load_truths_for_batch(l, state, b);
			const std::vector<BranchPoint> points = collect_branch_points(l, state, branch, b);
			std::vector<Assignment> assignments = assign_branch_targets(l, state, points, truths, b);

			float target_scores_sum = 0.0f;
			for (const Assignment & assignment : assignments)
			{
				target_scores_sum += assignment.target_score;
			}
			target_scores_sum = std::max(1.0f, target_scores_sum);

			const float cls_scale = branch_weight * l.cls_normalizer / target_scores_sum;
			for (std::size_t point_idx = 0; point_idx < points.size(); ++point_idx)
			{
				const BranchPoint & point = points[point_idx];
				Darknet::Layer & from = state.net.layers[point.input_index];
				const Assignment & assignment = assignments[point_idx];
				const int assigned_class = assignment.truth_idx >= 0 ? truths[assignment.truth_idx].class_id : -1;
				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const float target = class_id == assigned_class ? assignment.target_score : 0.0f;
					const float logit = class_logit(from, b, point.x, point.y, class_id, l.reg_max);
					branch_loss += cls_scale * bce_with_logits(logit, target);
					add_class_delta(from, b, point.x, point.y, class_id, l.classes, l.reg_max, target, cls_scale);
				}
			}

			for (std::size_t point_idx = 0; point_idx < points.size(); ++point_idx)
			{
				const Assignment & assignment = assignments[point_idx];
				if (assignment.truth_idx < 0 or assignment.target_score <= 0.0f)
				{
					continue;
				}

				const BranchPoint & point = points[point_idx];
				const GroundTruth & truth = truths[assignment.truth_idx];
				Darknet::Layer & from = state.net.layers[point.input_index];
				const float box_scale = branch_weight * l.box_normalizer * assignment.target_score / target_scores_sum;
				const float dfl_scale = branch_weight * l.dfl_normalizer * assignment.target_score / target_scores_sum;

				const float target_distances[4] =
				{
					point.anchor_x - truth.x1 / point.stride,
					point.anchor_y - truth.y1 / point.stride,
					truth.x2 / point.stride - point.anchor_x,
					truth.y2 / point.stride - point.anchor_y
				};

				branch_loss += add_ciou_projection_delta(from, l, point, b, truth, box_scale, state.net.w, state.net.h);
				for (int side = 0; side < 4; ++side)
				{
					branch_loss += add_dfl_delta(from, b, point.x, point.y, side, l.reg_max, target_distances[side], dfl_scale);
				}
			}
		}

		return branch_loss;
	}

	static void train_yolov9_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		if (state.truth == nullptr)
		{
			return;
		}

		zero_input_deltas(l, state);
		float loss = 0.0f;
		for (int branch = 0; branch < l.branch_count; ++branch)
		{
			loss += train_yolov9_branch(l, state, branch);
		}
		l.cost[0] = loss;
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


float yolov9_dfl_cross_entropy_delta(const float * logits, const int reg_max, const float target_distance, const float scale, float * delta)
{
	float probs[64];
	if (reg_max <= 0 or reg_max > static_cast<int>(sizeof(probs) / sizeof(probs[0])))
	{
		darknet_fatal_error(DARKNET_LOC, "YOLOv9 reg_max=%d exceeds local DFL buffer", reg_max);
	}
	softmax_bins(logits, reg_max, probs);

	const float clipped = std::max(0.0f, std::min(target_distance, static_cast<float>(reg_max) - 1.01f));
	const int left_bin = static_cast<int>(std::floor(clipped));
	const int right_bin = std::min(left_bin + 1, reg_max - 1);
	const float right_weight = clipped - left_bin;
	const float left_weight = 1.0f - right_weight;

	float loss = 0.0f;
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
		if (delta)
		{
			delta[bin] = scale * (target - probs[bin]);
		}
		if (target > 0.0f)
		{
			loss += -target * std::log(std::max(probs[bin], 1.0e-12f));
		}
	}

	return scale * loss;
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

	for (int i = 0; i < l.total; ++i)
	{
		Darknet::Layer & from = state.net.layers[l.input_layers[i]];
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
		for (int i = 0; i < l.total; ++i)
		{
			Darknet::Layer & from = state.net.layers[l.input_layers[i]];
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
