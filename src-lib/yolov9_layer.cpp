#include "darknet_internal.hpp"
#include "yolov9_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <vector>

float yolov9_dfl_cross_entropy_delta(const float *logits, int reg_max, float target_distance, float scale, float *delta);

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	constexpr int YOLOV9_GPU_META_STRIDE = 7;
	constexpr int YOLOV9_GPU_META_OUTPUTS = 0;
	constexpr int YOLOV9_GPU_META_W = 1;
	constexpr int YOLOV9_GPU_META_H = 2;
	constexpr int YOLOV9_GPU_META_C = 3;
	constexpr int YOLOV9_GPU_META_STRIDE_VALUE = 4;
	constexpr int YOLOV9_GPU_META_POINT_OFFSET = 5;
	constexpr int YOLOV9_GPU_META_OUTPUT_OFFSET = 6;

	static inline int yolov9_channel_index(const Darknet::Layer & from, const int b, const int c, const int y, const int x)
	{
		return b * from.outputs + c * from.out_h * from.out_w + y * from.out_w + x;
	}

	static inline float sigmoid(const float x)
	{
		return logistic_activate(x);
	}

	static inline float threshold_to_logit(const float threshold)
	{
		const float clipped = std::max(1.0e-6f, std::min(1.0f - 1.0e-6f, threshold));
		return std::log(clipped / (1.0f - clipped));
	}

	static inline int selected_branch_offset(const Darknet::Layer & l)
	{
		return l.inference_branch * l.n;
	}

	struct InferenceHeadView
	{
		const Darknet::Layer *combined = nullptr;
		const Darknet::Layer *box = nullptr;
		const Darknet::Layer *cls = nullptr;
		bool split = false;
	};

	static InferenceHeadView make_inference_head_view(const Darknet::Network * net, const Darknet::Layer & l, const int slot)
	{
		InferenceHeadView view;
		const int input_index = l.input_layers[slot];
		const Darknet::Layer & input = net->layers[input_index];
		if (input.type == Darknet::ELayerType::ROUTE and input.inference_direct_yolov9_head and input.n == 2 and input.input_layers != nullptr)
		{
			view.box = &net->layers[input.input_layers[0]];
			view.cls = &net->layers[input.input_layers[1]];
			view.split = true;
			return view;
		}
		view.combined = &input;
		return view;
	}

	static inline int head_out_w(const InferenceHeadView & view)
	{
		return view.split ? view.box->out_w : view.combined->out_w;
	}

	static inline int head_out_h(const InferenceHeadView & view)
	{
		return view.split ? view.box->out_h : view.combined->out_h;
	}

	static inline int head_spatial(const InferenceHeadView & view)
	{
		return head_out_w(view) * head_out_h(view);
	}

	static inline float head_box_logit(const InferenceHeadView & view, const int b, const int channel, const int y, const int x)
	{
		const Darknet::Layer & from = view.split ? *view.box : *view.combined;
		return from.output[yolov9_channel_index(from, b, channel, y, x)];
	}

	static inline float head_class_logit(const InferenceHeadView & view, const int b, const int class_id, const int reg_max, const int y, const int x)
	{
		if (view.split)
		{
			return view.cls->output[yolov9_channel_index(*view.cls, b, class_id, y, x)];
		}
		const int channel = 4 * reg_max + class_id;
		return view.combined->output[yolov9_channel_index(*view.combined, b, channel, y, x)];
	}

	static inline void require_yolov9_ciou_loss(const IOU_LOSS iou_loss)
	{
		if (iou_loss != CIOU)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 currently supports iou_loss=ciou only, got loss id #%d", static_cast<int>(iou_loss));
		}
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

	struct PixelBox
	{
		float x1 = 0.0f;
		float y1 = 0.0f;
		float x2 = 0.0f;
		float y2 = 0.0f;
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

	struct BatchAssignments
	{
		std::vector<GroundTruth> truths;
		std::vector<BranchPoint> points;
		std::vector<Assignment> assignments;
	};

	static inline PixelBox box_to_pixel_box(const Darknet::Box & box, const int netw, const int neth)
	{
		PixelBox result;
		result.x1 = (box.x - box.w * 0.5f) * netw;
		result.y1 = (box.y - box.h * 0.5f) * neth;
		result.x2 = (box.x + box.w * 0.5f) * netw;
		result.y2 = (box.y + box.h * 0.5f) * neth;
		return result;
	}

	static inline float pixel_box_ciou(const PixelBox & a, const PixelBox & b)
	{
		const float inter_x1 = std::max(a.x1, b.x1);
		const float inter_y1 = std::max(a.y1, b.y1);
		const float inter_x2 = std::min(a.x2, b.x2);
		const float inter_y2 = std::min(a.y2, b.y2);
		const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
		const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
		const float intersection = inter_w * inter_h;

		const float a_w = std::max(0.0f, a.x2 - a.x1);
		const float a_h = std::max(0.0f, a.y2 - a.y1);
		const float b_w = std::max(0.0f, b.x2 - b.x1);
		const float b_h = std::max(0.0f, b.y2 - b.y1);
		const float union_area = a_w * a_h + b_w * b_h - intersection;
		if (union_area <= 0.0f)
		{
			return 0.0f;
		}

		const float iou = intersection / union_area;
		const float a_cx = (a.x1 + a.x2) * 0.5f;
		const float a_cy = (a.y1 + a.y2) * 0.5f;
		const float b_cx = (b.x1 + b.x2) * 0.5f;
		const float b_cy = (b.y1 + b.y2) * 0.5f;
		const float center_distance = (a_cx - b_cx) * (a_cx - b_cx) + (a_cy - b_cy) * (a_cy - b_cy);
		const float convex_w = std::max(a.x2, b.x2) - std::min(a.x1, b.x1);
		const float convex_h = std::max(a.y2, b.y2) - std::min(a.y1, b.y1);
		const float convex_distance = convex_w * convex_w + convex_h * convex_h + 1.0e-9f;

		const float safe_a_w = std::max(a_w, 1.0e-9f);
		const float safe_a_h = std::max(a_h, 1.0e-9f);
		const float safe_b_w = std::max(b_w, 1.0e-9f);
		const float safe_b_h = std::max(b_h, 1.0e-9f);
		const float v = 4.0f / (M_PI * M_PI) * std::pow(std::atan(safe_b_w / safe_b_h) - std::atan(safe_a_w / safe_a_h), 2.0f);
		const float alpha = v / (1.0f - iou + v + 1.0e-9f);
		return iou - center_distance / convex_distance - alpha * v;
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

	static inline void get_dfl_distances(const InferenceHeadView & view, const int b, const int x, const int y, const int reg_max, float distances[4])
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
				logits[bin] = head_box_logit(view, b, channel, y, x);
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

	static inline Darknet::Box decode_prediction(const InferenceHeadView & view, const int b, const int x, const int y, const int stride, const int reg_max, const int netw, const int neth)
	{
		float distances[4];
		get_dfl_distances(view, b, x, y, reg_max, distances);
		return yolov9_dist2bbox(x + 0.5f, y + 0.5f, distances, static_cast<float>(stride), netw, neth);
	}

	static inline PixelBox decode_prediction_pixels(const Darknet::Layer & from, const int b, const int x, const int y, const int stride, const int reg_max, const int netw, const int neth)
	{
		return box_to_pixel_box(decode_prediction(from, b, x, y, stride, reg_max, netw, neth), netw, neth);
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

	static inline int best_class_logit(const Darknet::Layer & from, const int b, const int x, const int y, const int classes, const int reg_max, float & best_logit)
	{
		best_logit = -FLT_MAX;
		int best_class = -1;
		for (int class_id = 0; class_id < classes; ++class_id)
		{
			const int channel = 4 * reg_max + class_id;
			const float logit = from.output[yolov9_channel_index(from, b, channel, y, x)];
			if (logit > best_logit)
			{
				best_logit = logit;
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
			const PixelBox pixel_box = box_to_pixel_box(box, state.net.w, state.net.h);
			truth.x1 = pixel_box.x1;
			truth.y1 = pixel_box.y1;
			truth.x2 = pixel_box.x2;
			truth.y2 = pixel_box.y2;
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
				const PixelBox decoded_box = decode_prediction_pixels(from, b, point.x, point.y, point.stride, l.reg_max, state.net.w, state.net.h);
				const PixelBox truth_box{truth.x1, truth.y1, truth.x2, truth.y2};
				const float overlap = std::max(0.0f, pixel_box_ciou(decoded_box, truth_box));
				if (overlap <= 0.0f or score <= 0.0f)
				{
					continue;
				}

				const float metric = std::pow(score, l.tal_alpha) * std::pow(overlap, l.tal_beta);
				if (not std::isfinite(metric))
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

	static void validate_yolov9_input_shapes_after_resize(const Darknet::Layer & l, const Darknet::Network & net)
	{
		if (l.input_layers == nullptr or l.input_sizes == nullptr or l.strides == nullptr)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 resize validation requires input layers, input sizes, and strides");
		}
		if (l.classes <= 0 or l.reg_max <= 0)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 resize validation found invalid classes=%d or reg_max=%d", l.classes, l.reg_max);
		}
		if (l.branch_count <= 0 or l.n <= 0 or l.total != l.branch_count * l.n)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 resize validation found invalid branches=%d, scales=%d, inputs=%d", l.branch_count, l.n, l.total);
		}
		if (l.inference_branch < 0 or l.inference_branch >= l.branch_count)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 inference_branch=%d is outside [0, %d)", l.inference_branch, l.branch_count);
		}

		const int expected_channels = l.classes + 4 * l.reg_max;
		for (int branch = 0; branch < l.branch_count; ++branch)
		{
			for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
			{
				const int input_slot = branch * l.n + scale_idx;
				const int input_index = l.input_layers[input_slot];
				if (input_index < 0 or input_index >= net.n)
				{
					darknet_fatal_error(DARKNET_LOC, "YOLOv9 input slot %d points to invalid layer #%d", input_slot, input_index);
				}

				const int stride = l.strides[scale_idx];
				if (stride <= 0)
				{
					darknet_fatal_error(DARKNET_LOC, "YOLOv9 stride for scale %d must be positive, got %d", scale_idx, stride);
				}

				const Darknet::Layer & from = net.layers[input_index];
				if (from.out_w <= 0 or from.out_h <= 0 or from.out_c != expected_channels)
				{
					darknet_fatal_error(
						DARKNET_LOC,
						"YOLOv9 input layer #%d at branch %d scale %d has %d x %d x %d output, expected positive dimensions and %d channels",
						input_index,
						branch,
						scale_idx,
						from.out_w,
						from.out_h,
						from.out_c,
						expected_channels);
				}
				if (from.outputs <= 0)
				{
					darknet_fatal_error(DARKNET_LOC, "YOLOv9 input layer #%d at branch %d scale %d has non-positive outputs=%d", input_index, branch, scale_idx, from.outputs);
				}

				const long long expected_outputs = static_cast<long long>(from.out_w) * static_cast<long long>(from.out_h) * static_cast<long long>(from.out_c);
				if (expected_outputs != from.outputs)
				{
					darknet_fatal_error(
						DARKNET_LOC,
						"YOLOv9 input layer #%d at branch %d scale %d has outputs=%d, expected %lld from %d x %d x %d",
						input_index,
						branch,
						scale_idx,
						from.outputs,
						expected_outputs,
						from.out_w,
						from.out_h,
						from.out_c);
				}

				const long long scaled_w = static_cast<long long>(from.out_w) * static_cast<long long>(stride);
				const long long scaled_h = static_cast<long long>(from.out_h) * static_cast<long long>(stride);
				if (scaled_w != net.w or scaled_h != net.h)
				{
					darknet_fatal_error(
						DARKNET_LOC,
						"YOLOv9 input layer #%d at branch %d scale %d has %d x %d with stride %d, which maps to %lld x %lld instead of net %d x %d",
						input_index,
						branch,
						scale_idx,
						from.out_w,
						from.out_h,
						stride,
						scaled_w,
						scaled_h,
						net.w,
						net.h);
				}
			}
		}
	}

	static float box_loss_from_distances(
		const BranchPoint & point,
		const float distances[4],
		const GroundTruth & truth,
		const IOU_LOSS iou_loss,
		const int netw,
		const int neth)
	{
		require_yolov9_ciou_loss(iou_loss);
		const Darknet::Box pred = yolov9_dist2bbox(point.anchor_x, point.anchor_y, distances, static_cast<float>(point.stride), netw, neth);
		const float overlap = pixel_box_ciou(box_to_pixel_box(pred, netw, neth), PixelBox{truth.x1, truth.y1, truth.x2, truth.y2});
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

		std::vector<BatchAssignments> batch_assignments(l.batch);
		float target_scores_sum = 0.0f;
		for (int b = 0; b < l.batch; ++b)
		{
			BatchAssignments & batch = batch_assignments[b];
			batch.truths = load_truths_for_batch(l, state, b);
			batch.points = collect_branch_points(l, state, branch, b);
			batch.assignments = assign_branch_targets(l, state, batch.points, batch.truths, b);
			for (const Assignment & assignment : batch.assignments)
			{
				target_scores_sum += assignment.target_score;
			}
		}
		target_scores_sum = std::max(1.0f, target_scores_sum);
		const float batch_scale = static_cast<float>(l.batch);

		for (int b = 0; b < l.batch; ++b)
		{
			const BatchAssignments & batch = batch_assignments[b];
			const std::vector<GroundTruth> & truths = batch.truths;
			const std::vector<BranchPoint> & points = batch.points;
			const std::vector<Assignment> & assignments = batch.assignments;
			const float cls_scale = branch_weight * l.cls_normalizer * batch_scale / target_scores_sum;
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
				const float box_scale = branch_weight * l.box_normalizer * batch_scale * assignment.target_score / target_scores_sum;
				const float dfl_scale = branch_weight * l.dfl_normalizer * batch_scale * assignment.target_score / target_scores_sum / 4.0f;

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
		require_yolov9_ciou_loss(l.iou_loss);
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

#ifdef DARKNET_GPU
	static void ensure_yolov9_gpu_metadata_buffers(Darknet::Layer & l);
	static void update_yolov9_gpu_input_metadata(Darknet::Layer & l, Darknet::NetworkState state);

	static bool yolov9_can_use_gpu_compact_output(const Darknet::Network * net, const Darknet::Layer & l)
	{
		if (std::getenv("DARKNET_DISABLE_YOLOV9_GPU_COMPACT") != nullptr or l.batch != 1 or
			l.layers_output_gpu == nullptr or l.input_sizes_gpu == nullptr)
		{
			return false;
		}

		const int branch_offset = selected_branch_offset(l);
		for (int i = 0; i < l.n; ++i)
		{
			const int slot = branch_offset + i;
			if (slot < 0 or slot >= l.total)
			{
				return false;
			}
			const Darknet::Layer & input = net->layers[l.input_layers[slot]];
			if (input.type == Darknet::ELayerType::ROUTE and input.inference_direct_yolov9_head)
			{
				return false;
			}
		}
		return true;
	}

	static void pull_yolov9_full_inference_outputs_cpu(Darknet::Network * net, Darknet::Layer & l)
	{
		const int branch_offset = selected_branch_offset(l);
		for (int i = 0; i < l.n; ++i)
		{
			Darknet::Layer & input = net->layers[l.input_layers[branch_offset + i]];
			if (input.type == Darknet::ELayerType::ROUTE and input.inference_direct_yolov9_head and input.n == 2 and input.input_layers != nullptr)
			{
				Darknet::Layer & box = net->layers[input.input_layers[0]];
				Darknet::Layer & cls = net->layers[input.input_layers[1]];
				cuda_pull_array_async(box.output_gpu, box.output, box.batch * box.outputs);
				cuda_pull_array_async(cls.output_gpu, cls.output, cls.batch * cls.outputs);
			}
			else
			{
				cuda_pull_array_async(input.output_gpu, input.output, input.batch * input.outputs);
			}
		}
		CHECK_CUDA(cudaPeekAtLastError());
		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		l.inference_cpu_outputs_valid = 1;
		l.yolov9_compact_valid = 0;
	}

	static void ensure_yolov9_inference_outputs_cpu(Darknet::Network * net, Darknet::Layer & l, const float thresh, const bool allow_compact)
	{
		if (cfg_and_state.gpu_index < 0)
		{
			return;
		}

		if (l.inference_cpu_outputs_valid)
		{
			if (not l.yolov9_compact_valid)
			{
				return;
			}
			if (allow_compact and std::fabs(l.yolov9_compact_thresh - thresh) <= 1.0e-6f)
			{
				return;
			}
		}

		if (allow_compact)
		{
			Darknet::NetworkState state = {};
			state.net = *net;
			ensure_yolov9_gpu_metadata_buffers(l);
			update_yolov9_gpu_input_metadata(l, state);
			if (yolov9_can_use_gpu_compact_output(net, l))
			{
				compact_yolov9_detections_gpu(l, net->w, net->h, threshold_to_logit(thresh));
				l.inference_cpu_outputs_valid = 1;
				return;
			}
		}

		pull_yolov9_full_inference_outputs_cpu(net, l);
	}

	static std::unordered_map<const Darknet::Layer*, uint64_t> yolov9_gpu_metadata_signatures;

	static inline void yolov9_hash_metadata_value(uint64_t & hash, const uint64_t value)
	{
		hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
	}

	static uint64_t yolov9_gpu_input_metadata_signature(const Darknet::Layer & l, const Darknet::NetworkState & state)
	{
		uint64_t hash = 1469598103934665603ULL;
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.batch));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.outputs));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.total));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.n));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.branch_count));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.inference_branch));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.classes));
		yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.reg_max));
		yolov9_hash_metadata_value(hash, reinterpret_cast<uintptr_t>(l.input_sizes_gpu));
		yolov9_hash_metadata_value(hash, reinterpret_cast<uintptr_t>(l.layers_output_gpu));
		yolov9_hash_metadata_value(hash, reinterpret_cast<uintptr_t>(l.layers_delta_gpu));

		for (int slot = 0; slot < l.total; ++slot)
		{
			const int input_index = l.input_layers[slot];
			const Darknet::Layer & from = state.net.layers[input_index];
			yolov9_hash_metadata_value(hash, static_cast<uint64_t>(input_index));
			yolov9_hash_metadata_value(hash, static_cast<uint64_t>(l.strides[slot % l.n]));
			yolov9_hash_metadata_value(hash, static_cast<uint64_t>(from.outputs));
			yolov9_hash_metadata_value(hash, static_cast<uint64_t>(from.out_w));
			yolov9_hash_metadata_value(hash, static_cast<uint64_t>(from.out_h));
			yolov9_hash_metadata_value(hash, static_cast<uint64_t>(from.out_c));
			yolov9_hash_metadata_value(hash, reinterpret_cast<uintptr_t>(from.output_gpu));
			yolov9_hash_metadata_value(hash, reinterpret_cast<uintptr_t>(from.delta_gpu));
		}
		return hash;
	}

	static void invalidate_yolov9_gpu_input_metadata_cache(const Darknet::Layer & l)
	{
		yolov9_gpu_metadata_signatures.erase(&l);
	}

	static size_t yolov9_gpu_workspace_floats(const Darknet::Layer & l)
	{
		const int channels = l.classes + 4 * l.reg_max;
		if (channels <= 0 or l.outputs <= 0)
		{
			return 0;
		}
		const int total_points = l.outputs / channels;
		const size_t point_count = static_cast<size_t>(l.batch) * static_cast<size_t>(total_points);
		const size_t class_count = point_count * static_cast<size_t>(l.classes);
		const size_t gt_slots = static_cast<size_t>(l.batch) * static_cast<size_t>(l.max_boxes);
		const size_t top_count = gt_slots * static_cast<size_t>(std::max(1, l.tal_topk));
		return point_count * 4		// decoded boxes
			+ class_count			// predicted class probabilities
			+ class_count			// target class scores
			+ point_count * 3		// assignment gt, overlap, metric
			+ top_count * 3		// top-k point, metric, overlap
			+ gt_slots * 2		// per-truth normalization maxima
			+ 8;				// loss/normalization stats
	}

	static void ensure_yolov9_gpu_metadata_buffers(Darknet::Layer & l)
	{
		bool allocated_metadata_buffer = false;
		if (l.input_sizes_gpu == nullptr)
		{
			l.input_sizes_gpu = cuda_make_int_array_new_api(nullptr, l.total * YOLOV9_GPU_META_STRIDE);
			allocated_metadata_buffer = true;
		}
		if (l.layers_output_gpu == nullptr)
		{
			l.layers_output_gpu = reinterpret_cast<float**>(cuda_make_array_pointers(nullptr, l.total));
			allocated_metadata_buffer = true;
		}
		if (l.layers_delta_gpu == nullptr)
		{
			l.layers_delta_gpu = reinterpret_cast<float**>(cuda_make_array_pointers(nullptr, l.total));
			allocated_metadata_buffer = true;
		}
		if (allocated_metadata_buffer)
		{
			invalidate_yolov9_gpu_input_metadata_cache(l);
		}
	}

	static void ensure_yolov9_gpu_buffers(Darknet::Layer & l)
	{
		ensure_yolov9_gpu_metadata_buffers(l);
		if (l.loss_gpu == nullptr)
		{
			l.loss_gpu = cuda_make_array(nullptr, yolov9_gpu_workspace_floats(l));
		}
	}

	static void update_yolov9_gpu_input_metadata(Darknet::Layer & l, Darknet::NetworkState state)
	{
		const uint64_t metadata_signature = yolov9_gpu_input_metadata_signature(l, state);
		const auto cached = yolov9_gpu_metadata_signatures.find(&l);
		if (cached != yolov9_gpu_metadata_signatures.end() and cached->second == metadata_signature)
		{
			return;
		}

		std::vector<int> metadata(l.total * YOLOV9_GPU_META_STRIDE, 0);
		std::vector<float*> outputs(l.total, nullptr);
		std::vector<float*> deltas(l.total, nullptr);
		std::vector<int> point_offsets(l.total, 0);
		std::vector<int> output_offsets(l.total, 0);

		for (int branch = 0; branch < l.branch_count; ++branch)
		{
			int point_offset = 0;
			int output_offset = 0;
			for (int scale = 0; scale < l.n; ++scale)
			{
				const int slot = branch * l.n + scale;
				const Darknet::Layer & from = state.net.layers[l.input_layers[slot]];
				point_offsets[slot] = point_offset;
				output_offsets[slot] = output_offset;
				point_offset += from.out_w * from.out_h;
				output_offset += from.outputs;
			}
		}

		for (int slot = 0; slot < l.total; ++slot)
		{
			const Darknet::Layer & from = state.net.layers[l.input_layers[slot]];
			int *m = metadata.data() + slot * YOLOV9_GPU_META_STRIDE;
			m[YOLOV9_GPU_META_OUTPUTS] = from.outputs;
			m[YOLOV9_GPU_META_W] = from.out_w;
			m[YOLOV9_GPU_META_H] = from.out_h;
			m[YOLOV9_GPU_META_C] = from.out_c;
			m[YOLOV9_GPU_META_STRIDE_VALUE] = l.strides[slot % l.n];
			m[YOLOV9_GPU_META_POINT_OFFSET] = point_offsets[slot];
			m[YOLOV9_GPU_META_OUTPUT_OFFSET] = output_offsets[slot];
			outputs[slot] = from.output_gpu;
			deltas[slot] = from.delta_gpu;
		}

		memcpy_ongpu(l.input_sizes_gpu, metadata.data(), metadata.size() * sizeof(int));
		memcpy_ongpu(l.layers_output_gpu, outputs.data(), outputs.size() * sizeof(float*));
		memcpy_ongpu(l.layers_delta_gpu, deltas.data(), deltas.size() * sizeof(float*));
		yolov9_gpu_metadata_signatures[&l] = metadata_signature;
	}

#endif
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
	l.iou_loss = CIOU;

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

	if (l == nullptr or net == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot resize YOLOv9 layer without a layer and network");
	}
	validate_yolov9_input_shapes_after_resize(*l, *net);

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
	l->inference_cpu_outputs_valid = 0;
	l->yolov9_compact_valid = 0;

	l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	if (l->loss_gpu)
	{
		cuda_free(l->loss_gpu);
		l->loss_gpu = nullptr;
	}
	invalidate_yolov9_gpu_input_metadata_cache(*l);
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

#ifdef DARKNET_GPU
	ensure_yolov9_inference_outputs_cpu(const_cast<Darknet::Network *>(net), const_cast<Darknet::Layer &>(l), thresh, false);
#endif

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	const float thresh_logit = threshold_to_logit(thresh);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const InferenceHeadView view = make_inference_head_view(net, l, branch_offset + scale_idx);
		for (int y = 0; y < head_out_h(view); ++y)
		{
			for (int x = 0; x < head_out_w(view); ++x)
			{
				float best_logit = -FLT_MAX;
				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const float logit = head_class_logit(view, 0, class_id, l.reg_max, y, x);
					if (logit > best_logit)
					{
						best_logit = logit;
					}
				}
				if (best_logit > thresh_logit)
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

	Darknet::Layer & l = net->layers[index];
#ifdef DARKNET_GPU
	ensure_yolov9_inference_outputs_cpu(net, l, thresh, true);
	if (l.yolov9_compact_valid)
	{
		for (int compact_index = 0; compact_index < l.yolov9_compact_count; ++compact_index)
		{
			const float *record = l.yolov9_compact_cpu + compact_index * l.yolov9_compact_stride;
			Darknet::Output_Object oo;
			oo.layer_index = index;
			oo.n = static_cast<int>(record[0]);
			oo.i = static_cast<int>(record[1]);
			oo.obj_index = static_cast<int>(record[2]);
			oo.compact_index = compact_index;
			cache.push_back(oo);
		}
		return l.yolov9_compact_count;
	}
#endif
	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	const float thresh_logit = threshold_to_logit(thresh);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const InferenceHeadView view = make_inference_head_view(net, l, branch_offset + scale_idx);
		for (int y = 0; y < head_out_h(view); ++y)
		{
			for (int x = 0; x < head_out_w(view); ++x)
			{
				float best_logit = -FLT_MAX;
				int best_class = -1;
				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const float logit = head_class_logit(view, 0, class_id, l.reg_max, y, x);
					if (logit > best_logit)
					{
						best_logit = logit;
						best_class = class_id;
					}
				}
				if (best_logit > thresh_logit)
				{
					Darknet::Output_Object oo;
					oo.layer_index = index;
					oo.n = scale_idx;
					oo.i = y * head_out_w(view) + x;
					oo.obj_index = best_class;
					oo.compact_index = -1;
					cache.push_back(oo);
					count++;
				}
			}
		}
	}
	return count;
}


int yolov9_num_detections_batch(const Darknet::Network * net, const Darknet::Layer & l, float thresh, int batch)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	ensure_yolov9_inference_outputs_cpu(const_cast<Darknet::Network *>(net), const_cast<Darknet::Layer &>(l), thresh, false);
#endif

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	const float thresh_logit = threshold_to_logit(thresh);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const InferenceHeadView view = make_inference_head_view(net, l, branch_offset + scale_idx);
		for (int y = 0; y < head_out_h(view); ++y)
		{
			for (int x = 0; x < head_out_w(view); ++x)
			{
				float best_logit = -FLT_MAX;
				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const float logit = head_class_logit(view, batch, class_id, l.reg_max, y, x);
					if (logit > best_logit)
					{
						best_logit = logit;
					}
				}
				if (best_logit > thresh_logit)
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

#ifdef DARKNET_GPU
	ensure_yolov9_inference_outputs_cpu(const_cast<Darknet::Network *>(net), const_cast<Darknet::Layer &>(l), thresh, false);
#endif

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	const float thresh_logit = threshold_to_logit(thresh);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const InferenceHeadView view = make_inference_head_view(net, l, branch_offset + scale_idx);
		const int stride = std::max(1, l.strides[scale_idx]);
		for (int y = 0; y < head_out_h(view); ++y)
		{
			for (int x = 0; x < head_out_w(view); ++x)
			{
				float best_logit = -FLT_MAX;
				int best_class = -1;
				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const float logit = head_class_logit(view, 0, class_id, l.reg_max, y, x);
					if (logit > best_logit)
					{
						best_logit = logit;
						best_class = class_id;
					}
				}
				if (best_logit <= thresh_logit)
				{
					continue;
				}

				dets[count].bbox = decode_prediction(view, 0, x, y, stride, l.reg_max, netw, neth);
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
					const float score = sigmoid(head_class_logit(view, 0, class_id, l.reg_max, y, x));
					dets[count].prob[mapped_class_id] = (score > thresh) ? score : 0.0f;
				}
				count++;
			}
		}
	}

	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
	return count;
}


int get_yolov9_detections_v3(Darknet::Network * net, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	for (const auto & oo : cache)
	{
		if (oo.layer_index >= 0 and oo.layer_index < net->n and net->layers[oo.layer_index].type == Darknet::ELayerType::YOLOV9)
		{
#ifdef DARKNET_GPU
			ensure_yolov9_inference_outputs_cpu(net, net->layers[oo.layer_index], thresh, true);
#endif
			break;
		}
	}

	int count = 0;
	const float thresh_logit = threshold_to_logit(thresh);
	for (const auto & oo : cache)
	{
		const Darknet::Layer & l = net->layers[oo.layer_index];
		if (l.type != Darknet::ELayerType::YOLOV9)
		{
			continue;
		}
		if (oo.compact_index >= 0 and l.yolov9_compact_valid and oo.compact_index < l.yolov9_compact_count)
		{
			const float *record = l.yolov9_compact_cpu + oo.compact_index * l.yolov9_compact_stride;
			dets[count].bbox.x = record[3];
			dets[count].bbox.y = record[4];
			dets[count].bbox.w = record[5];
			dets[count].bbox.h = record[6];
			dets[count].objectness = 1.0f;
			dets[count].classes = l.classes;
			dets[count].best_class_idx = static_cast<int>(record[2]);
			for (int class_id = 0; class_id < l.classes; ++class_id)
			{
				const int mapped_class_id = map ? map[class_id] : class_id;
				if (mapped_class_id < 0 or mapped_class_id >= l.classes)
				{
					continue;
				}
				dets[count].prob[mapped_class_id] = record[7 + class_id];
			}
			count++;
			continue;
		}
		const int branch_offset = selected_branch_offset(l);
		if (oo.n < 0 or oo.n >= l.n)
		{
			continue;
		}
		const InferenceHeadView view = make_inference_head_view(net, l, branch_offset + oo.n);
		const int spatial = head_spatial(view);
		if (oo.i < 0 or oo.i >= spatial)
		{
			continue;
		}
		const int x = oo.i % head_out_w(view);
		const int y = oo.i / head_out_w(view);
		const int stride = std::max(1, l.strides[oo.n]);
		int best_class = oo.obj_index;
		float best_logit = -FLT_MAX;
		if (best_class >= 0 and best_class < l.classes)
		{
			best_logit = head_class_logit(view, 0, best_class, l.reg_max, y, x);
		}
		else
		{
			for (int class_id = 0; class_id < l.classes; ++class_id)
			{
				const float logit = head_class_logit(view, 0, class_id, l.reg_max, y, x);
				if (logit > best_logit)
				{
					best_logit = logit;
					best_class = class_id;
				}
			}
		}
		if (best_class < 0 or best_logit <= thresh_logit)
		{
			continue;
		}

		dets[count].bbox = decode_prediction(view, 0, x, y, stride, l.reg_max, netw, neth);
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
			const float score = sigmoid(head_class_logit(view, 0, class_id, l.reg_max, y, x));
			dets[count].prob[mapped_class_id] = (score > thresh) ? score : 0.0f;
		}
		count++;
	}

	correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
	return count;
}


int get_yolov9_detections_batch(const Darknet::Network * net, const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter, int batch)
{
	TAT(TATPARMS);

	int count = 0;
	const int branch_offset = selected_branch_offset(l);
	const float thresh_logit = threshold_to_logit(thresh);
	for (int scale_idx = 0; scale_idx < l.n; ++scale_idx)
	{
		const InferenceHeadView view = make_inference_head_view(net, l, branch_offset + scale_idx);
		const int stride = std::max(1, l.strides[scale_idx]);
		for (int y = 0; y < head_out_h(view); ++y)
		{
			for (int x = 0; x < head_out_w(view); ++x)
			{
				float best_logit = -FLT_MAX;
				int best_class = -1;
				for (int class_id = 0; class_id < l.classes; ++class_id)
				{
					const float logit = head_class_logit(view, batch, class_id, l.reg_max, y, x);
					if (logit > best_logit)
					{
						best_logit = logit;
						best_class = class_id;
					}
				}
				if (best_logit <= thresh_logit)
				{
					continue;
				}

				dets[count].bbox = decode_prediction(view, batch, x, y, stride, l.reg_max, netw, neth);
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
					const float score = sigmoid(head_class_logit(view, batch, class_id, l.reg_max, y, x));
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

	const bool force_cpu_fallback = std::getenv("DARKNET_YOLOV9_FORCE_CPU_FALLBACK") != nullptr;
	const bool can_use_native_gpu_training =
		state.train and
		not l.onlyforward and
		state.truth != nullptr and
		(l.branch_count == 1 or l.branch_count == 2) and
		not force_cpu_fallback;

	if (can_use_native_gpu_training)
	{
		static bool logged_native_gpu_path = false;
		if (not logged_native_gpu_path)
		{
			*cfg_and_state.output << "YOLOv9: using native CUDA DDetect/DualDDetect TAL training path." << std::endl;
			logged_native_gpu_path = true;
		}
		require_yolov9_ciou_loss(l.iou_loss);
		ensure_yolov9_gpu_buffers(l);
		update_yolov9_gpu_input_metadata(l, state);
		train_yolov9_single_branch_gpu(l, state);
		return;
	}

	if (not state.train or l.onlyforward)
	{
		const bool pack_inference_output = std::getenv("DARKNET_YOLOV9_PACK_INFERENCE_OUTPUT") != nullptr;
		const bool pull_in_forward = std::getenv("DARKNET_YOLOV9_PULL_IN_FORWARD") != nullptr;
		l.inference_cpu_outputs_valid = 0;
		l.yolov9_compact_valid = 0;
		if (pack_inference_output or pull_in_forward)
		{
			ensure_yolov9_inference_outputs_cpu(&state.net, l, 0.0f, false);
		}

		if (pack_inference_output)
		{
			Darknet::NetworkState cpu_state = state;
			cpu_state.truth = nullptr;
			cpu_state.train = 0;
			forward_yolov9_layer(l, cpu_state);
			cuda_push_array(l.output_gpu, l.output, l.batch * l.outputs);
		}
		return;
	}

	if (state.train and not l.onlyforward)
	{
		static bool logged_cpu_fallback = false;
		if (not logged_cpu_fallback)
		{
			*cfg_and_state.output << "YOLOv9: using CPU fallback for GPU training path"
				<< " (branches=" << l.branch_count
				<< ", force_cpu_fallback=" << (force_cpu_fallback ? "yes" : "no")
				<< ")." << std::endl;
			logged_cpu_fallback = true;
		}
	}

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

	if (state.train and not l.onlyforward)
	{
		return;
	}

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
