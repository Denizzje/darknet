#include "darknet_internal.hpp"

#include <algorithm>
#include <cfloat>

namespace
{
	constexpr int YOLOV9_META_STRIDE = 7;
	constexpr int YOLOV9_META_OUTPUTS = 0;
	constexpr int YOLOV9_META_W = 1;
	constexpr int YOLOV9_META_H = 2;
	constexpr int YOLOV9_META_STRIDE_VALUE = 4;
	constexpr int YOLOV9_META_POINT_OFFSET = 5;
	constexpr float YOLOV9_PI = 3.14159265358979323846f;

	__device__ float yolov9_sigmoid_gpu(const float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}

	__device__ float yolov9_bce_with_logits_gpu(const float logit, const float target)
	{
		const float max_value = fmaxf(logit, 0.0f);
		return max_value - logit * target + log1pf(expf(-fabsf(logit)));
	}

	__device__ int yolov9_find_scale_gpu(const int point, const int scale_count, const int *meta)
	{
		for (int scale = 0; scale < scale_count; ++scale)
		{
			const int *m = meta + scale * YOLOV9_META_STRIDE;
			const int offset = m[YOLOV9_META_POINT_OFFSET];
			const int spatial = m[YOLOV9_META_W] * m[YOLOV9_META_H];
			if (point >= offset and point < offset + spatial)
			{
				return scale;
			}
		}
		return scale_count - 1;
	}

	__device__ int yolov9_channel_index_gpu(
		const int outputs,
		const int w,
		const int h,
		const int b,
		const int channel,
		const int y,
		const int x)
	{
		return b * outputs + channel * h * w + y * w + x;
	}

	__device__ void yolov9_point_location_gpu(
		const int point,
		const int scale_count,
		const int *meta,
		int *scale,
		int *x,
		int *y,
		int *stride)
	{
		const int s = yolov9_find_scale_gpu(point, scale_count, meta);
		const int *m = meta + s * YOLOV9_META_STRIDE;
		const int local = point - m[YOLOV9_META_POINT_OFFSET];
		const int w = m[YOLOV9_META_W];
		*scale = s;
		*x = local % w;
		*y = local / w;
		*stride = m[YOLOV9_META_STRIDE_VALUE];
	}

	__device__ float yolov9_ciou_pixels_gpu(
		const float ax1,
		const float ay1,
		const float ax2,
		const float ay2,
		const float bx1,
		const float by1,
		const float bx2,
		const float by2)
	{
		const float inter_x1 = fmaxf(ax1, bx1);
		const float inter_y1 = fmaxf(ay1, by1);
		const float inter_x2 = fminf(ax2, bx2);
		const float inter_y2 = fminf(ay2, by2);
		const float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
		const float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
		const float intersection = inter_w * inter_h;

		const float a_w = fmaxf(0.0f, ax2 - ax1);
		const float a_h = fmaxf(0.0f, ay2 - ay1);
		const float b_w = fmaxf(0.0f, bx2 - bx1);
		const float b_h = fmaxf(0.0f, by2 - by1);
		const float union_area = a_w * a_h + b_w * b_h - intersection;
		if (union_area <= 0.0f)
		{
			return 0.0f;
		}

		const float iou = intersection / union_area;
		const float a_cx = (ax1 + ax2) * 0.5f;
		const float a_cy = (ay1 + ay2) * 0.5f;
		const float b_cx = (bx1 + bx2) * 0.5f;
		const float b_cy = (by1 + by2) * 0.5f;
		const float center_distance = (a_cx - b_cx) * (a_cx - b_cx) + (a_cy - b_cy) * (a_cy - b_cy);
		const float convex_w = fmaxf(ax2, bx2) - fminf(ax1, bx1);
		const float convex_h = fmaxf(ay2, by2) - fminf(ay1, by1);
		const float convex_distance = convex_w * convex_w + convex_h * convex_h + 1.0e-9f;

		const float safe_a_w = fmaxf(a_w, 1.0e-9f);
		const float safe_a_h = fmaxf(a_h, 1.0e-9f);
		const float safe_b_w = fmaxf(b_w, 1.0e-9f);
		const float safe_b_h = fmaxf(b_h, 1.0e-9f);
		const float v = 4.0f / (YOLOV9_PI * YOLOV9_PI) * powf(atanf(safe_b_w / safe_b_h) - atanf(safe_a_w / safe_a_h), 2.0f);
		const float alpha = v / (1.0f - iou + v + 1.0e-9f);
		return iou - center_distance / convex_distance - alpha * v;
	}

	__device__ float yolov9_atomic_max_float(float *address, const float value)
	{
		int *address_as_int = reinterpret_cast<int*>(address);
		int old = *address_as_int;
		while (value > __int_as_float(old))
		{
			const int assumed = old;
			old = atomicCAS(address_as_int, assumed, __float_as_int(value));
			if (assumed == old)
			{
				break;
			}
		}
		return __int_as_float(old);
	}

	__device__ void yolov9_truth_pixels_gpu(
		const float *truth,
		const int truth_size,
		const int truths_per_batch,
		const int b,
		const int t,
		const int netw,
		const int neth,
		float *x1,
		float *y1,
		float *x2,
		float *y2,
		int *class_id,
		bool *valid)
	{
		const float *truth_ptr = truth + b * truths_per_batch + t * truth_size;
		const float x = truth_ptr[0];
		const float y = truth_ptr[1];
		const float w = truth_ptr[2];
		const float h = truth_ptr[3];
		*class_id = static_cast<int>(truth_ptr[4]);
		*valid = x > 0.0f and w > 0.0f and h > 0.0f;
		*x1 = (x - w * 0.5f) * netw;
		*y1 = (y - h * 0.5f) * neth;
		*x2 = (x + w * 0.5f) * netw;
		*y2 = (y + h * 0.5f) * neth;
	}

	__device__ void yolov9_softmax_project_side_gpu(
		const float *input,
		const int index_base,
		const int channel_step,
		const int reg_max,
		float *distance,
		float probs[64])
	{
		float max_logit = input[index_base];
		for (int bin = 1; bin < reg_max; ++bin)
		{
			max_logit = fmaxf(max_logit, input[index_base + bin * channel_step]);
		}

		float sum = 0.0f;
		for (int bin = 0; bin < reg_max; ++bin)
		{
			const float value = expf(input[index_base + bin * channel_step] - max_logit);
			probs[bin] = value;
			sum += value;
		}

		const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
		float projected = 0.0f;
		for (int bin = 0; bin < reg_max; ++bin)
		{
			probs[bin] *= inv_sum;
			projected += static_cast<float>(bin) * probs[bin];
		}
		*distance = projected;
	}

	__device__ float yolov9_box_loss_from_distances_gpu(
		const float anchor_x,
		const float anchor_y,
		const float stride,
		const float distances[4],
		const float truth_x1,
		const float truth_y1,
		const float truth_x2,
		const float truth_y2)
	{
		const float pred_x1 = (anchor_x - distances[0]) * stride;
		const float pred_y1 = (anchor_y - distances[1]) * stride;
		const float pred_x2 = (anchor_x + distances[2]) * stride;
		const float pred_y2 = (anchor_y + distances[3]) * stride;
		return 1.0f - yolov9_ciou_pixels_gpu(pred_x1, pred_y1, pred_x2, pred_y2, truth_x1, truth_y1, truth_x2, truth_y2);
	}

	__global__ void yolov9_decode_predictions_kernel(
		const int n,
		const int total_points,
		const int classes,
		const int reg_max,
		const int scale_count,
		float **inputs,
		const int *meta,
		float *pred_boxes,
		float *pred_scores)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int point = id % total_points;
		const int b = id / total_points;
		int scale = 0;
		int x = 0;
		int y = 0;
		int stride = 1;
		yolov9_point_location_gpu(point, scale_count, meta, &scale, &x, &y, &stride);
		const int *m = meta + scale * YOLOV9_META_STRIDE;
		const int outputs = m[YOLOV9_META_OUTPUTS];
		const int w = m[YOLOV9_META_W];
		const int h = m[YOLOV9_META_H];
		const int spatial = w * h;
		const float *input = inputs[scale];
		const int spatial_index = y * w + x;

		float distances[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		float probs[64];
		for (int side = 0; side < 4; ++side)
		{
			const int channel = side * reg_max;
			const int index = yolov9_channel_index_gpu(outputs, w, h, b, channel, y, x);
			yolov9_softmax_project_side_gpu(input, index, spatial, reg_max, &distances[side], probs);
		}

		const float anchor_x = static_cast<float>(x) + 0.5f;
		const float anchor_y = static_cast<float>(y) + 0.5f;
		const int box_index = (b * total_points + point) * 4;
		pred_boxes[box_index + 0] = (anchor_x - distances[0]) * stride;
		pred_boxes[box_index + 1] = (anchor_y - distances[1]) * stride;
		pred_boxes[box_index + 2] = (anchor_x + distances[2]) * stride;
		pred_boxes[box_index + 3] = (anchor_y + distances[3]) * stride;

		const int score_index = (b * total_points + point) * classes;
		for (int class_id = 0; class_id < classes; ++class_id)
		{
			const int channel = 4 * reg_max + class_id;
			const int input_index = b * outputs + channel * spatial + spatial_index;
			pred_scores[score_index + class_id] = yolov9_sigmoid_gpu(input[input_index]);
		}
	}

	__global__ void yolov9_compact_detections_kernel(
		const int n,
		const int total_points,
		const int classes,
		const int reg_max,
		const int netw,
		const int neth,
		const float thresh_logit,
		const int record_stride,
		const int capacity,
		const int scale_count,
		float **inputs,
		const int *meta,
		int *compact_count,
		float *compact_records)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n or reg_max <= 0 or reg_max > 64)
		{
			return;
		}

		const int point = id % total_points;
		const int b = id / total_points;
		int scale = 0;
		int x = 0;
		int y = 0;
		int stride = 1;
		yolov9_point_location_gpu(point, scale_count, meta, &scale, &x, &y, &stride);
		const int *m = meta + scale * YOLOV9_META_STRIDE;
		const int outputs = m[YOLOV9_META_OUTPUTS];
		const int w = m[YOLOV9_META_W];
		const int h = m[YOLOV9_META_H];
		const int spatial = w * h;
		const int spatial_index = y * w + x;
		const float *input = inputs[scale];

		float best_logit = -FLT_MAX;
		int best_class = -1;
		for (int class_id = 0; class_id < classes; ++class_id)
		{
			const int channel = 4 * reg_max + class_id;
			const int input_index = b * outputs + channel * spatial + spatial_index;
			const float logit = input[input_index];
			if (logit > best_logit)
			{
				best_logit = logit;
				best_class = class_id;
			}
		}
		if (best_class < 0 or best_logit <= thresh_logit)
		{
			return;
		}

		const int record_index = atomicAdd(compact_count, 1);
		if (record_index >= capacity)
		{
			return;
		}

		float distances[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		float probs[64];
		for (int side = 0; side < 4; ++side)
		{
			const int channel = side * reg_max;
			const int input_index = yolov9_channel_index_gpu(outputs, w, h, b, channel, y, x);
			yolov9_softmax_project_side_gpu(input, input_index, spatial, reg_max, &distances[side], probs);
		}

		const float anchor_x = static_cast<float>(x) + 0.5f;
		const float anchor_y = static_cast<float>(y) + 0.5f;
		const float x1 = (anchor_x - distances[0]) * static_cast<float>(stride);
		const float y1 = (anchor_y - distances[1]) * static_cast<float>(stride);
		const float x2 = (anchor_x + distances[2]) * static_cast<float>(stride);
		const float y2 = (anchor_y + distances[3]) * static_cast<float>(stride);

		float *record = compact_records + record_index * record_stride;
		record[0] = static_cast<float>(scale);
		record[1] = static_cast<float>(spatial_index);
		record[2] = static_cast<float>(best_class);
		record[3] = ((x1 + x2) * 0.5f) / static_cast<float>(netw);
		record[4] = ((y1 + y2) * 0.5f) / static_cast<float>(neth);
		record[5] = (x2 - x1) / static_cast<float>(netw);
		record[6] = (y2 - y1) / static_cast<float>(neth);

		for (int class_id = 0; class_id < classes; ++class_id)
		{
			const int channel = 4 * reg_max + class_id;
			const int input_index = b * outputs + channel * spatial + spatial_index;
			const float score = yolov9_sigmoid_gpu(input[input_index]);
			record[7 + class_id] = score > yolov9_sigmoid_gpu(thresh_logit) ? score : 0.0f;
		}
	}

	__global__ void yolov9_topk_kernel(
		const int n,
		const int total_points,
		const int classes,
		const int max_boxes,
		const int truth_size,
		const int truths_per_batch,
		const int netw,
		const int neth,
		const int topk,
		const float tal_alpha,
		const float tal_beta,
		const int scale_count,
		const int *meta,
		const float *truth,
		const float *pred_boxes,
		const float *pred_scores,
		float *top_indices,
		float *top_metrics,
		float *top_overlaps)
	{
		const int gt_slot = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (gt_slot >= n)
		{
			return;
		}

		const int b = gt_slot / max_boxes;
		const int t = gt_slot % max_boxes;
		for (int k = 0; k < topk; ++k)
		{
			const int top_offset = gt_slot * topk + k;
			top_indices[top_offset] = -1.0f;
			top_metrics[top_offset] = 0.0f;
			top_overlaps[top_offset] = 0.0f;
		}

		float truth_x1 = 0.0f;
		float truth_y1 = 0.0f;
		float truth_x2 = 0.0f;
		float truth_y2 = 0.0f;
		int class_id = -1;
		bool valid = false;
		yolov9_truth_pixels_gpu(truth, truth_size, truths_per_batch, b, t, netw, neth, &truth_x1, &truth_y1, &truth_x2, &truth_y2, &class_id, &valid);
		if (not valid or class_id < 0 or class_id >= classes)
		{
			return;
		}

		for (int point = 0; point < total_points; ++point)
		{
			int scale = 0;
			int x = 0;
			int y = 0;
			int stride = 1;
			yolov9_point_location_gpu(point, scale_count, meta, &scale, &x, &y, &stride);
			const float anchor_x = (static_cast<float>(x) + 0.5f) * stride;
			const float anchor_y = (static_cast<float>(y) + 0.5f) * stride;
			if (not (anchor_x > truth_x1 and anchor_y > truth_y1 and anchor_x < truth_x2 and anchor_y < truth_y2))
			{
				continue;
			}

			const int box_index = (b * total_points + point) * 4;
			const float overlap = fmaxf(
				0.0f,
				yolov9_ciou_pixels_gpu(
					pred_boxes[box_index + 0],
					pred_boxes[box_index + 1],
					pred_boxes[box_index + 2],
					pred_boxes[box_index + 3],
					truth_x1,
					truth_y1,
					truth_x2,
					truth_y2));
			if (overlap <= 0.0f)
			{
				continue;
			}

			const float score = pred_scores[(b * total_points + point) * classes + class_id];
			if (score <= 0.0f)
			{
				continue;
			}

			const float metric = powf(score, tal_alpha) * powf(overlap, tal_beta);
			if (not isfinite(metric))
			{
				continue;
			}

			int min_k = 0;
			float min_metric = top_metrics[gt_slot * topk];
			for (int k = 1; k < topk; ++k)
			{
				const float existing = top_metrics[gt_slot * topk + k];
				if (existing < min_metric)
				{
					min_metric = existing;
					min_k = k;
				}
			}
			if (metric > min_metric)
			{
				const int top_offset = gt_slot * topk + min_k;
				top_indices[top_offset] = static_cast<float>(point);
				top_metrics[top_offset] = metric;
				top_overlaps[top_offset] = overlap;
			}
		}
	}

	__global__ void yolov9_resolve_overlap_batched_kernel(
		const int n,
		const int total_points,
		const int max_boxes,
		const int topk,
		const float *top_indices,
		const float *top_overlaps,
		float *assignment_overlap)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int candidate = id % topk;
		const int gt_slot = id / topk;
		const int b = gt_slot / max_boxes;
		const int point = static_cast<int>(top_indices[gt_slot * topk + candidate]);
		if (point < 0)
		{
			return;
		}
		const float overlap = top_overlaps[gt_slot * topk + candidate];
		yolov9_atomic_max_float(assignment_overlap + b * total_points + point, overlap);
	}

	__global__ void yolov9_write_assignments_kernel(
		const int n,
		const int total_points,
		const int max_boxes,
		const int topk,
		const float *top_indices,
		const float *top_metrics,
		const float *top_overlaps,
		float *assignment_gt,
		float *assignment_metric,
		const float *assignment_overlap)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int candidate = id % topk;
		const int gt_slot = id / topk;
		const int b = gt_slot / max_boxes;
		const int t = gt_slot % max_boxes;
		const int point = static_cast<int>(top_indices[gt_slot * topk + candidate]);
		if (point < 0)
		{
			return;
		}
		const int assignment_index = b * total_points + point;
		const float overlap = top_overlaps[gt_slot * topk + candidate];
		if (fabsf(overlap - assignment_overlap[assignment_index]) <= 1.0e-7f)
		{
			assignment_gt[assignment_index] = static_cast<float>(t);
			assignment_metric[assignment_index] = top_metrics[gt_slot * topk + candidate];
		}
	}

	__global__ void yolov9_truth_max_kernel(
		const int n,
		const int total_points,
		const int max_boxes,
		const int topk,
		const float *top_indices,
		const float *top_metrics,
		const float *top_overlaps,
		const float *assignment_gt,
		float *max_metric_by_truth,
		float *max_overlap_by_truth)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int candidate = id % topk;
		const int gt_slot = id / topk;
		const int b = gt_slot / max_boxes;
		const int t = gt_slot % max_boxes;
		const int point = static_cast<int>(top_indices[gt_slot * topk + candidate]);
		if (point < 0)
		{
			return;
		}
		const int assignment_index = b * total_points + point;
		if (static_cast<int>(assignment_gt[assignment_index]) != t)
		{
			return;
		}
		yolov9_atomic_max_float(max_metric_by_truth + gt_slot, top_metrics[gt_slot * topk + candidate]);
		yolov9_atomic_max_float(max_overlap_by_truth + gt_slot, top_overlaps[gt_slot * topk + candidate]);
	}

	__global__ void yolov9_target_scores_kernel(
		const int n,
		const int total_points,
		const int classes,
		const int max_boxes,
		const int truth_size,
		const int truths_per_batch,
		const float *truth,
		const float *assignment_gt,
		const float *assignment_metric,
		const float *max_metric_by_truth,
		const float *max_overlap_by_truth,
		float *target_scores,
		float *loss_stats)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int b = id / total_points;
		const int point = id % total_points;
		const int t = static_cast<int>(assignment_gt[id]);
		if (t < 0)
		{
			return;
		}

		const int gt_slot = b * max_boxes + t;
		const float max_metric = max_metric_by_truth[gt_slot];
		const float max_overlap = max_overlap_by_truth[gt_slot];
		if (max_metric <= 0.0f)
		{
			return;
		}

		const int class_id = static_cast<int>(truth[b * truths_per_batch + t * truth_size + 4]);
		if (class_id < 0 or class_id >= classes)
		{
			return;
		}

		const float target_score = fminf(1.0f, assignment_metric[id] * max_overlap / (max_metric + 1.0e-9f));
		if (target_score <= 0.0f)
		{
			return;
		}
		target_scores[(b * total_points + point) * classes + class_id] = target_score;
		atomicAdd(loss_stats + 0, target_score);
	}

	__global__ void yolov9_class_loss_delta_kernel(
		const int n,
		const int total_points,
		const int classes,
		const int reg_max,
		const int scale_count,
		const float cls_normalizer,
		const float loss_scale,
		float **inputs,
		float **input_deltas,
		const int *meta,
		const float *pred_scores,
		const float *target_scores,
		float *loss_stats)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int class_id = id % classes;
		const int point_linear = id / classes;
		const int point = point_linear % total_points;
		const int b = point_linear / total_points;
		int scale = 0;
		int x = 0;
		int y = 0;
		int stride = 1;
		yolov9_point_location_gpu(point, scale_count, meta, &scale, &x, &y, &stride);
		(void)stride;
		const int *m = meta + scale * YOLOV9_META_STRIDE;
		const int outputs = m[YOLOV9_META_OUTPUTS];
		const int w = m[YOLOV9_META_W];
		const int h = m[YOLOV9_META_H];
		const int channel = 4 * reg_max + class_id;
		const int input_index = yolov9_channel_index_gpu(outputs, w, h, b, channel, y, x);
		const float target_sum = fmaxf(1.0f, loss_stats[0]);
		const float batch_scale = cls_normalizer * static_cast<float>(n / (total_points * classes)) / target_sum;
		const float target = target_scores[id];
		const float logit = inputs[scale][input_index];
		atomicAdd(loss_stats + 2, batch_scale * yolov9_bce_with_logits_gpu(logit, target));
		input_deltas[scale][input_index] += loss_scale * batch_scale * (target - pred_scores[id]);
	}

	__global__ void yolov9_box_dfl_loss_delta_kernel(
		const int n,
		const int total_points,
		const int classes,
		const int reg_max,
		const int max_boxes,
		const int truth_size,
		const int truths_per_batch,
		const int netw,
		const int neth,
		const int scale_count,
		const float box_normalizer,
		const float dfl_normalizer,
		const float loss_scale,
		float **inputs,
		float **input_deltas,
		const int *meta,
		const float *truth,
		const float *assignment_gt,
		const float *target_scores,
		float *loss_stats)
	{
		const int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (id >= n)
		{
			return;
		}

		const int b = id / total_points;
		const int point = id % total_points;
		const int t = static_cast<int>(assignment_gt[id]);
		if (t < 0)
		{
			return;
		}

		float truth_x1 = 0.0f;
		float truth_y1 = 0.0f;
		float truth_x2 = 0.0f;
		float truth_y2 = 0.0f;
		int class_id = -1;
		bool valid = false;
		yolov9_truth_pixels_gpu(truth, truth_size, truths_per_batch, b, t, netw, neth, &truth_x1, &truth_y1, &truth_x2, &truth_y2, &class_id, &valid);
		if (not valid or class_id < 0 or class_id >= classes)
		{
			return;
		}

		const float target_score = target_scores[(b * total_points + point) * classes + class_id];
		if (target_score <= 0.0f)
		{
			return;
		}

		int scale = 0;
		int x = 0;
		int y = 0;
		int stride = 1;
		yolov9_point_location_gpu(point, scale_count, meta, &scale, &x, &y, &stride);
		const int *m = meta + scale * YOLOV9_META_STRIDE;
		const int outputs = m[YOLOV9_META_OUTPUTS];
		const int w = m[YOLOV9_META_W];
		const int h = m[YOLOV9_META_H];
		const int spatial = w * h;
		const float anchor_x = static_cast<float>(x) + 0.5f;
		const float anchor_y = static_cast<float>(y) + 0.5f;
		float *input = inputs[scale];
		float *delta = input_deltas[scale];
		const float target_sum = fmaxf(1.0f, loss_stats[0]);
		const float batch = static_cast<float>(n / total_points);
		const float box_scale = box_normalizer * batch * target_score / target_sum;
		const float dfl_scale = dfl_normalizer * batch * target_score / target_sum / 4.0f;

		float distances[4];
		float side_probs[4][64];
		for (int side = 0; side < 4; ++side)
		{
			const int channel = side * reg_max;
			const int input_index = yolov9_channel_index_gpu(outputs, w, h, b, channel, y, x);
			yolov9_softmax_project_side_gpu(input, input_index, spatial, reg_max, &distances[side], side_probs[side]);
		}

		const float base_loss = yolov9_box_loss_from_distances_gpu(anchor_x, anchor_y, static_cast<float>(stride), distances, truth_x1, truth_y1, truth_x2, truth_y2);
		atomicAdd(loss_stats + 3, box_scale * base_loss);

		const float eps = 0.05f;
		for (int side = 0; side < 4; ++side)
		{
			float plus_distances[4] = {distances[0], distances[1], distances[2], distances[3]};
			float minus_distances[4] = {distances[0], distances[1], distances[2], distances[3]};
			plus_distances[side] = fminf(static_cast<float>(reg_max) - 1.0f, distances[side] + eps);
			minus_distances[side] = fmaxf(0.0f, distances[side] - eps);
			const float actual_eps = plus_distances[side] - minus_distances[side];
			if (actual_eps > 0.0f)
			{
				const float loss_plus = yolov9_box_loss_from_distances_gpu(anchor_x, anchor_y, static_cast<float>(stride), plus_distances, truth_x1, truth_y1, truth_x2, truth_y2);
				const float loss_minus = yolov9_box_loss_from_distances_gpu(anchor_x, anchor_y, static_cast<float>(stride), minus_distances, truth_x1, truth_y1, truth_x2, truth_y2);
				const float dloss_ddistance = (loss_plus - loss_minus) / actual_eps;
				for (int bin = 0; bin < reg_max; ++bin)
				{
					const float dproject_dlogit = side_probs[side][bin] * (static_cast<float>(bin) - distances[side]);
					const int channel = side * reg_max + bin;
					const int input_index = yolov9_channel_index_gpu(outputs, w, h, b, channel, y, x);
					delta[input_index] += -loss_scale * box_scale * dloss_ddistance * dproject_dlogit;
				}
			}

			const float target_distances[4] =
			{
				anchor_x - truth_x1 / static_cast<float>(stride),
				anchor_y - truth_y1 / static_cast<float>(stride),
				truth_x2 / static_cast<float>(stride) - anchor_x,
				truth_y2 / static_cast<float>(stride) - anchor_y
			};
			const float clipped = fmaxf(0.0f, fminf(target_distances[side], static_cast<float>(reg_max) - 1.01f));
			const int left_bin = static_cast<int>(floorf(clipped));
			const int right_bin = (left_bin + 1 < reg_max - 1) ? left_bin + 1 : reg_max - 1;
			const float right_weight = clipped - static_cast<float>(left_bin);
			const float left_weight = 1.0f - right_weight;
			float dfl_loss = 0.0f;
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
				if (target > 0.0f)
				{
					dfl_loss += -target * logf(fmaxf(side_probs[side][bin], 1.0e-12f));
				}
				const int channel = side * reg_max + bin;
				const int input_index = yolov9_channel_index_gpu(outputs, w, h, b, channel, y, x);
				delta[input_index] += loss_scale * dfl_scale * (target - side_probs[side][bin]);
			}
			atomicAdd(loss_stats + 4, dfl_scale * dfl_loss);
		}
	}
}


int compact_yolov9_detections_gpu(Darknet::Layer & l, int netw, int neth, float thresh_logit)
{
	TAT(TATPARMS);

	const int total_points = l.outputs / (l.classes + 4 * l.reg_max);
	const int capacity = l.batch * total_points;
	const int record_stride = 7 + l.classes;
	if (capacity <= 0 or record_stride <= 7)
	{
		l.yolov9_compact_valid = 1;
		l.yolov9_compact_count = 0;
		l.yolov9_compact_thresh = 1.0f / (1.0f + expf(-thresh_logit));
		return 0;
	}

	if (l.yolov9_compact_capacity != capacity or l.yolov9_compact_stride != record_stride or
		l.yolov9_compact_cpu == nullptr or l.yolov9_compact_gpu == nullptr or l.yolov9_compact_count_gpu == nullptr)
	{
		if (l.yolov9_compact_cpu)
		{
			free(l.yolov9_compact_cpu);
			l.yolov9_compact_cpu = nullptr;
		}
		if (l.yolov9_compact_gpu)
		{
			cuda_free(l.yolov9_compact_gpu);
			l.yolov9_compact_gpu = nullptr;
		}
		if (l.yolov9_compact_count_gpu)
		{
			cuda_free(reinterpret_cast<float *>(l.yolov9_compact_count_gpu));
			l.yolov9_compact_count_gpu = nullptr;
		}
		l.yolov9_compact_cpu = static_cast<float *>(xcalloc(static_cast<size_t>(capacity) * record_stride, sizeof(float)));
		l.yolov9_compact_gpu = cuda_make_array(nullptr, static_cast<size_t>(capacity) * record_stride);
		l.yolov9_compact_count_gpu = cuda_make_int_array_new_api(nullptr, 1);
		l.yolov9_compact_capacity = capacity;
		l.yolov9_compact_stride = record_stride;
	}

	CHECK_CUDA(cudaMemsetAsync(l.yolov9_compact_count_gpu, 0, sizeof(int), get_cuda_stream()));
	const int branch = std::max(0, std::min(l.inference_branch, l.branch_count - 1));
	const int branch_offset = branch * l.n;
	yolov9_compact_detections_kernel<<<cuda_gridsize(capacity), BLOCK, 0, get_cuda_stream()>>>(
		capacity,
		total_points,
		l.classes,
		l.reg_max,
		netw,
		neth,
		thresh_logit,
		record_stride,
		capacity,
		l.n,
		l.layers_output_gpu + branch_offset,
		l.input_sizes_gpu + branch_offset * YOLOV9_META_STRIDE,
		l.yolov9_compact_count_gpu,
		l.yolov9_compact_gpu);
	CHECK_CUDA(cudaPeekAtLastError());

	int count = 0;
	CHECK_CUDA(cudaMemcpyAsync(&count, l.yolov9_compact_count_gpu, sizeof(int), cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
	count = std::max(0, std::min(count, capacity));
	if (count > 0)
	{
		cuda_pull_array(l.yolov9_compact_gpu, l.yolov9_compact_cpu, static_cast<size_t>(count) * record_stride);
	}
	l.yolov9_compact_count = count;
	l.yolov9_compact_valid = 1;
	l.yolov9_compact_thresh = 1.0f / (1.0f + expf(-thresh_logit));
	return count;
}


void train_yolov9_single_branch_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int batch = l.batch;
	const int total_points = l.outputs / (l.classes + 4 * l.reg_max);
	const int point_count = batch * total_points;
	const int class_count = point_count * l.classes;
	const int gt_slots = batch * l.max_boxes;
	const int topk = std::max(1, l.tal_topk);
	const int top_count = gt_slots * topk;

	float *work = l.loss_gpu;
	float *pred_boxes = work;
	float *pred_scores = pred_boxes + point_count * 4;
	float *target_scores = pred_scores + class_count;
	float *assignment_gt = target_scores + class_count;
	float *assignment_overlap = assignment_gt + point_count;
	float *assignment_metric = assignment_overlap + point_count;
	float *top_indices = assignment_metric + point_count;
	float *top_metrics = top_indices + top_count;
	float *top_overlaps = top_metrics + top_count;
	float *max_metric_by_truth = top_overlaps + top_count;
	float *max_overlap_by_truth = max_metric_by_truth + gt_slots;
	float *loss_stats = max_overlap_by_truth + gt_slots;
	float total_loss = 0.0f;

	for (int slot = 0; slot < l.total; ++slot)
	{
		fill_ongpu(l.input_sizes[slot] * batch, 0.0f, state.net.layers[l.input_layers[slot]].delta_gpu, 1);
	}
	fill_ongpu(l.outputs * batch, 0.0f, l.delta_gpu, 1);

	for (int branch = 0; branch < l.branch_count; ++branch)
	{
		const int branch_offset = branch * l.n;
		const float branch_weight = (branch == l.inference_branch) ? 1.0f : l.aux_loss_weight;
		float **branch_outputs = l.layers_output_gpu + branch_offset;
		float **branch_deltas = l.layers_delta_gpu + branch_offset;
		int *branch_meta = l.input_sizes_gpu + branch_offset * YOLOV9_META_STRIDE;

		fill_ongpu(class_count, 0.0f, target_scores, 1);
		fill_ongpu(point_count, -1.0f, assignment_gt, 1);
		fill_ongpu(point_count, 0.0f, assignment_overlap, 1);
		fill_ongpu(point_count, 0.0f, assignment_metric, 1);
		fill_ongpu(gt_slots, 0.0f, max_metric_by_truth, 1);
		fill_ongpu(gt_slots, 0.0f, max_overlap_by_truth, 1);
		fill_ongpu(8, 0.0f, loss_stats, 1);

		yolov9_decode_predictions_kernel<<<cuda_gridsize(point_count), BLOCK, 0, get_cuda_stream()>>>(
			point_count,
			total_points,
			l.classes,
			l.reg_max,
			l.n,
			branch_outputs,
			branch_meta,
			pred_boxes,
			pred_scores);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_topk_kernel<<<cuda_gridsize(gt_slots), BLOCK, 0, get_cuda_stream()>>>(
			gt_slots,
			total_points,
			l.classes,
			l.max_boxes,
			l.truth_size,
			l.truths,
			state.net.w,
			state.net.h,
			topk,
			l.tal_alpha,
			l.tal_beta,
			l.n,
			branch_meta,
			state.truth,
			pred_boxes,
			pred_scores,
			top_indices,
			top_metrics,
			top_overlaps);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_resolve_overlap_batched_kernel<<<cuda_gridsize(top_count), BLOCK, 0, get_cuda_stream()>>>(
			top_count,
			total_points,
			l.max_boxes,
			topk,
			top_indices,
			top_overlaps,
			assignment_overlap);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_write_assignments_kernel<<<cuda_gridsize(top_count), BLOCK, 0, get_cuda_stream()>>>(
			top_count,
			total_points,
			l.max_boxes,
			topk,
			top_indices,
			top_metrics,
			top_overlaps,
			assignment_gt,
			assignment_metric,
			assignment_overlap);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_truth_max_kernel<<<cuda_gridsize(top_count), BLOCK, 0, get_cuda_stream()>>>(
			top_count,
			total_points,
			l.max_boxes,
			topk,
			top_indices,
			top_metrics,
			top_overlaps,
			assignment_gt,
			max_metric_by_truth,
			max_overlap_by_truth);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_target_scores_kernel<<<cuda_gridsize(point_count), BLOCK, 0, get_cuda_stream()>>>(
			point_count,
			total_points,
			l.classes,
			l.max_boxes,
			l.truth_size,
			l.truths,
			state.truth,
			assignment_gt,
			assignment_metric,
			max_metric_by_truth,
			max_overlap_by_truth,
			target_scores,
			loss_stats);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_class_loss_delta_kernel<<<cuda_gridsize(class_count), BLOCK, 0, get_cuda_stream()>>>(
			class_count,
			total_points,
			l.classes,
			l.reg_max,
			l.n,
			branch_weight * l.cls_normalizer,
			state.net.loss_scale,
			branch_outputs,
			branch_deltas,
			branch_meta,
			pred_scores,
			target_scores,
			loss_stats);
		CHECK_CUDA(cudaPeekAtLastError());

		yolov9_box_dfl_loss_delta_kernel<<<cuda_gridsize(point_count), BLOCK, 0, get_cuda_stream()>>>(
			point_count,
			total_points,
			l.classes,
			l.reg_max,
			l.max_boxes,
			l.truth_size,
			l.truths,
			state.net.w,
			state.net.h,
			l.n,
			branch_weight * l.box_normalizer,
			branch_weight * l.dfl_normalizer,
			state.net.loss_scale,
			branch_outputs,
			branch_deltas,
			branch_meta,
			state.truth,
			assignment_gt,
			target_scores,
			loss_stats);
		CHECK_CUDA(cudaPeekAtLastError());

		float loss_cpu[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		cuda_pull_array(loss_stats, loss_cpu, 5);
		total_loss += loss_cpu[2] + loss_cpu[3] + loss_cpu[4];
	}
	l.cost[0] = total_loss;
}
