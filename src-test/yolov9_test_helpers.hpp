#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace YoloV9Test
{
	struct Point
	{
		float x = 0.0f;
		float y = 0.0f;
	};

	struct FeatureMapShape
	{
		int width = 0;
		int height = 0;
		float stride = 0.0f;
	};

	struct AnchorGrid
	{
		std::vector<Point> points;
		std::vector<float> strides;
	};

	struct BoxXYXY
	{
		float x1 = 0.0f;
		float y1 = 0.0f;
		float x2 = 0.0f;
		float y2 = 0.0f;
	};

	struct BoxXYWH
	{
		float x = 0.0f;
		float y = 0.0f;
		float w = 0.0f;
		float h = 0.0f;
	};

	struct DecodedPrediction
	{
		BoxXYWH box;
		std::vector<float> class_scores;
	};

	struct NoObjectnessDetection
	{
		BoxXYWH box;
		float objectness = 0.0f;
		std::vector<float> probabilities;
	};

	struct DDetectShapeConfig
	{
		int batch = 0;
		int classes = 0;
		int reg_max = 0;
		std::vector<FeatureMapShape> levels;
	};

	struct DDetectShapeSummary
	{
		int logits_channels_per_point = 0;
		int decoded_channels_per_point = 0;
		std::size_t total_points = 0;
		std::vector<std::size_t> points_per_level;
		std::vector<std::array<int, 4>> training_tensor_shapes;
		std::array<int, 3> inference_tensor_shape = {0, 0, 0};
	};

	std::vector<float> dfl_project_one_anchor(const std::vector<float> & logits, int reg_max);
	AnchorGrid make_anchor_points(const std::vector<FeatureMapShape> & levels, float grid_cell_offset = 0.5f);
	BoxXYXY dist2bbox_xyxy(const Point & anchor, const std::array<float, 4> & ltrb);
	BoxXYWH dist2bbox_xywh(const Point & anchor, const std::array<float, 4> & ltrb);
	std::vector<NoObjectnessDetection> extract_no_objectness_detections(
		const std::vector<DecodedPrediction> & predictions,
		float threshold);
	DDetectShapeSummary summarize_ddetect_shapes(const DDetectShapeConfig & config);
}
