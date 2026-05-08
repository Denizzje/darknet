#include "yolov9_test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
	void require_positive(const int value, const char * name)
	{
		if (value <= 0)
		{
			throw std::invalid_argument(std::string(name) + " must be positive");
		}
	}
}

namespace YoloV9Test
{
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
}
