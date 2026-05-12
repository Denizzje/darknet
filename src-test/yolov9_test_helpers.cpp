#include "yolov9_test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
	void require_positive(const int value, const char * name)
	{
		if (value <= 0)
		{
			throw std::invalid_argument(std::string(name) + " must be positive");
		}
	}

	void require_finite_non_negative(const float value, const char * name)
	{
		if (not std::isfinite(value) or value < 0.0f)
		{
			throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
		}
	}
}

namespace YoloV9Test
{
	namespace
	{
		struct RankedTalCandidate
		{
			int truth_idx = -1;
			int point_idx = -1;
			float metric = 0.0f;
			float overlap = 0.0f;
		};
	}

	std::vector<float> dfl_project_one_anchor(const std::vector<float> & logits, const int reg_max)
	{
		require_positive(reg_max, "reg_max");
		if (logits.size() != static_cast<std::size_t>(4 * reg_max))
		{
			throw std::invalid_argument("DFL logits must contain 4 * reg_max values for one anchor point");
		}

		std::vector<float> distances(4, 0.0f);
		for (int side = 0; side < 4; ++side)
		{
			const auto offset = static_cast<std::size_t>(side * reg_max);
			float max_logit = -std::numeric_limits<float>::infinity();
			for (int bin = 0; bin < reg_max; ++bin)
			{
				max_logit = std::max(max_logit, logits[offset + static_cast<std::size_t>(bin)]);
			}

			float denominator = 0.0f;
			float numerator = 0.0f;
			for (int bin = 0; bin < reg_max; ++bin)
			{
				const float weight = std::exp(logits[offset + static_cast<std::size_t>(bin)] - max_logit);
				denominator += weight;
				numerator += static_cast<float>(bin) * weight;
			}

			if (denominator == 0.0f or not std::isfinite(denominator))
			{
				throw std::invalid_argument("DFL softmax denominator must be finite and non-zero");
			}
			distances[static_cast<std::size_t>(side)] = numerator / denominator;
		}

		return distances;
	}

	AnchorGrid make_anchor_points(const std::vector<FeatureMapShape> & levels, const float grid_cell_offset)
	{
		if (levels.empty())
		{
			throw std::invalid_argument("at least one feature map level is required");
		}

		AnchorGrid grid;
		for (const auto & level : levels)
		{
			require_positive(level.width, "feature map width");
			require_positive(level.height, "feature map height");
			if (level.stride <= 0.0f)
			{
				throw std::invalid_argument("feature map stride must be positive");
			}

			const auto points = static_cast<std::size_t>(level.width) * static_cast<std::size_t>(level.height);
			grid.points.reserve(grid.points.size() + points);
			grid.strides.reserve(grid.strides.size() + points);

			for (int y = 0; y < level.height; ++y)
			{
				for (int x = 0; x < level.width; ++x)
				{
					grid.points.push_back(Point{static_cast<float>(x) + grid_cell_offset, static_cast<float>(y) + grid_cell_offset});
					grid.strides.push_back(level.stride);
				}
			}
		}

		return grid;
	}

	BoxXYXY dist2bbox_xyxy(const Point & anchor, const std::array<float, 4> & ltrb)
	{
		return BoxXYXY
		{
			anchor.x - ltrb[0],
			anchor.y - ltrb[1],
			anchor.x + ltrb[2],
			anchor.y + ltrb[3]
		};
	}

	BoxXYWH dist2bbox_xywh(const Point & anchor, const std::array<float, 4> & ltrb)
	{
		const auto xyxy = dist2bbox_xyxy(anchor, ltrb);
		return BoxXYWH
		{
			(xyxy.x1 + xyxy.x2) * 0.5f,
			(xyxy.y1 + xyxy.y2) * 0.5f,
			xyxy.x2 - xyxy.x1,
			xyxy.y2 - xyxy.y1
		};
	}

	std::vector<NoObjectnessDetection> extract_no_objectness_detections(
		const std::vector<DecodedPrediction> & predictions,
		const float threshold)
	{
		if (threshold < 0.0f or threshold > 1.0f)
		{
			throw std::invalid_argument("threshold must be in [0, 1]");
		}

		std::vector<NoObjectnessDetection> detections;
		for (const auto & prediction : predictions)
		{
			if (prediction.class_scores.empty())
			{
				throw std::invalid_argument("YOLOv9 predictions must provide class scores");
			}

			NoObjectnessDetection detection;
			detection.box = prediction.box;
			detection.objectness = 1.0f;
			detection.probabilities.assign(prediction.class_scores.size(), 0.0f);

			bool keep = false;
			for (std::size_t class_index = 0; class_index < prediction.class_scores.size(); ++class_index)
			{
				const float score = prediction.class_scores[class_index];
				if (score >= threshold)
				{
					detection.probabilities[class_index] = score;
					keep = true;
				}
			}

			if (keep)
			{
				detections.push_back(detection);
			}
		}

		return detections;
	}

	DDetectShapeSummary summarize_ddetect_shapes(const DDetectShapeConfig & config)
	{
		require_positive(config.batch, "batch");
		require_positive(config.classes, "classes");
		require_positive(config.reg_max, "reg_max");
		if (config.levels.empty())
		{
			throw std::invalid_argument("at least one DDetect feature level is required");
		}

		DDetectShapeSummary summary;
		summary.logits_channels_per_point = config.classes + 4 * config.reg_max;
		summary.decoded_channels_per_point = 4 + config.classes;
		summary.points_per_level.reserve(config.levels.size());
		summary.training_tensor_shapes.reserve(config.levels.size());

		for (const auto & level : config.levels)
		{
			require_positive(level.width, "feature map width");
			require_positive(level.height, "feature map height");
			const auto points = static_cast<std::size_t>(level.width) * static_cast<std::size_t>(level.height);
			summary.points_per_level.push_back(points);
			summary.total_points += points;
			summary.training_tensor_shapes.push_back(
				{config.batch, summary.logits_channels_per_point, level.height, level.width});
		}

		summary.inference_tensor_shape =
			{config.batch, summary.decoded_channels_per_point, static_cast<int>(summary.total_points)};

		return summary;
	}

	std::vector<TalAssignment> assign_tal_targets(
		const TalAssignerConfig & config,
		const std::vector<TalCandidate> & candidates)
	{
		require_positive(config.point_count, "point_count");
		require_positive(config.truth_count, "truth_count");
		if (config.topk <= 0)
		{
			throw std::invalid_argument("topk must be positive");
		}
		require_finite_non_negative(config.alpha, "alpha");
		require_finite_non_negative(config.beta, "beta");

		std::vector<TalAssignment> assignments(static_cast<std::size_t>(config.point_count));
		std::vector<std::vector<RankedTalCandidate>> topk_per_truth(static_cast<std::size_t>(config.truth_count));
		for (const TalCandidate & candidate : candidates)
		{
			if (candidate.truth_idx < 0 or candidate.truth_idx >= config.truth_count)
			{
				throw std::invalid_argument("candidate truth_idx is outside the configured truth range");
			}
			if (candidate.point_idx < 0 or candidate.point_idx >= config.point_count)
			{
				throw std::invalid_argument("candidate point_idx is outside the configured point range");
			}
			if (not candidate.anchor_inside_truth or candidate.class_score <= 0.0f or candidate.overlap <= 0.0f)
			{
				continue;
			}
			if (not std::isfinite(candidate.class_score) or not std::isfinite(candidate.overlap))
			{
				continue;
			}

			const float metric = std::pow(candidate.class_score, config.alpha) * std::pow(candidate.overlap, config.beta);
			if (not std::isfinite(metric))
			{
				continue;
			}

			RankedTalCandidate assignment;
			assignment.truth_idx = candidate.truth_idx;
			assignment.point_idx = candidate.point_idx;
			assignment.metric = metric;
			assignment.overlap = candidate.overlap;
			topk_per_truth[static_cast<std::size_t>(candidate.truth_idx)].push_back(assignment);
		}

		const int topk = std::max(1, config.topk);
		for (std::vector<RankedTalCandidate> & truth_candidates : topk_per_truth)
		{
			std::sort(
				truth_candidates.begin(),
				truth_candidates.end(),
				[](const RankedTalCandidate & lhs, const RankedTalCandidate & rhs)
				{
					return lhs.metric > rhs.metric;
				});
			if (static_cast<int>(truth_candidates.size()) > topk)
			{
				truth_candidates.resize(static_cast<std::size_t>(topk));
			}
		}

		for (const std::vector<RankedTalCandidate> & truth_candidates : topk_per_truth)
		{
			for (const RankedTalCandidate & candidate : truth_candidates)
			{
				TalAssignment & current = assignments[static_cast<std::size_t>(candidate.point_idx)];
				if (current.truth_idx < 0 or candidate.overlap > current.overlap)
				{
					current.truth_idx = candidate.truth_idx;
					current.metric = candidate.metric;
					current.overlap = candidate.overlap;
					current.target_score = 0.0f;
				}
			}
		}

		std::vector<float> max_metric_by_truth(static_cast<std::size_t>(config.truth_count), 0.0f);
		std::vector<float> max_overlap_by_truth(static_cast<std::size_t>(config.truth_count), 0.0f);
		for (const TalAssignment & assignment : assignments)
		{
			if (assignment.truth_idx < 0)
			{
				continue;
			}
			const auto truth_idx = static_cast<std::size_t>(assignment.truth_idx);
			max_metric_by_truth[truth_idx] = std::max(max_metric_by_truth[truth_idx], assignment.metric);
			max_overlap_by_truth[truth_idx] = std::max(max_overlap_by_truth[truth_idx], assignment.overlap);
		}

		for (TalAssignment & assignment : assignments)
		{
			if (assignment.truth_idx < 0)
			{
				continue;
			}
			const auto truth_idx = static_cast<std::size_t>(assignment.truth_idx);
			const float max_metric = max_metric_by_truth[truth_idx];
			const float max_overlap = max_overlap_by_truth[truth_idx];
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

	float normalized_target_scores_sum(const std::vector<TalAssignment> & assignments)
	{
		float sum = 0.0f;
		for (const TalAssignment & assignment : assignments)
		{
			if (assignment.truth_idx >= 0)
			{
				sum += assignment.target_score;
			}
		}
		return std::max(1.0f, sum);
	}

	BranchLossScales branch_loss_scales(
		const BranchLossScaleConfig & config,
		const int branch,
		const float target_score)
	{
		require_positive(config.branch_count, "branch_count");
		if (config.inference_branch < 0 or config.inference_branch >= config.branch_count)
		{
			throw std::invalid_argument("inference_branch is outside the configured branch range");
		}
		if (branch < 0 or branch >= config.branch_count)
		{
			throw std::invalid_argument("branch is outside the configured branch range");
		}
		require_finite_non_negative(config.aux_loss_weight, "aux_loss_weight");
		require_finite_non_negative(config.batch_size, "batch_size");
		require_finite_non_negative(config.target_scores_sum, "target_scores_sum");
		require_finite_non_negative(config.cls_normalizer, "cls_normalizer");
		require_finite_non_negative(config.box_normalizer, "box_normalizer");
		require_finite_non_negative(config.dfl_normalizer, "dfl_normalizer");
		require_finite_non_negative(target_score, "target_score");
		if (config.batch_size <= 0.0f)
		{
			throw std::invalid_argument("batch_size must be positive");
		}

		const float denominator = std::max(1.0f, config.target_scores_sum);
		BranchLossScales scales;
		scales.branch_weight = branch == config.inference_branch ? 1.0f : config.aux_loss_weight;
		scales.cls_scale = scales.branch_weight * config.cls_normalizer * config.batch_size / denominator;
		scales.box_scale = scales.branch_weight * config.box_normalizer * config.batch_size * target_score / denominator;
		scales.dfl_scale = scales.branch_weight * config.dfl_normalizer * config.batch_size * target_score / denominator / 4.0f;
		return scales;
	}

	YoloV9ResizeShapeSummary summarize_yolov9_resize_shapes(const YoloV9ResizeShapeConfig & config)
	{
		require_positive(config.net_width, "net_width");
		require_positive(config.net_height, "net_height");
		require_positive(config.classes, "classes");
		require_positive(config.reg_max, "reg_max");
		require_positive(config.branch_count, "branch_count");
		if (config.inputs.empty())
		{
			throw std::invalid_argument("at least one YOLOv9 input shape is required");
		}
		if (config.inputs.size() % static_cast<std::size_t>(config.branch_count) != 0)
		{
			throw std::invalid_argument("YOLOv9 input count must be divisible by branch_count");
		}

		const int scale_count = static_cast<int>(config.inputs.size()) / config.branch_count;
		if (config.inference_branch < 0 or config.inference_branch >= config.branch_count)
		{
			throw std::invalid_argument("inference_branch is outside the configured branch range");
		}

		YoloV9ResizeShapeSummary summary;
		summary.expected_channels = config.classes + 4 * config.reg_max;
		summary.input_sizes.reserve(config.inputs.size());

		for (std::size_t idx = 0; idx < config.inputs.size(); ++idx)
		{
			const YoloV9InputShape & input = config.inputs[idx];
			require_positive(input.width, "input width");
			require_positive(input.height, "input height");
			require_positive(input.channels, "input channels");
			require_positive(input.stride, "input stride");
			if (input.channels != summary.expected_channels)
			{
				throw std::invalid_argument("YOLOv9 input channels must match classes + 4 * reg_max");
			}
			if (input.width * input.stride != config.net_width or input.height * input.stride != config.net_height)
			{
				throw std::invalid_argument("YOLOv9 input stride must map feature dimensions back to network dimensions");
			}

			const int input_size = input.width * input.height * input.channels;
			summary.input_sizes.push_back(input_size);
			summary.aggregate_inputs += input_size;
			const int branch = static_cast<int>(idx) / scale_count;
			if (branch == config.inference_branch)
			{
				summary.inference_outputs += input_size;
			}
		}

		return summary;
	}
}
