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

	struct TalAssignerConfig
	{
		int point_count = 0;
		int truth_count = 0;
		int topk = 10;
		float alpha = 0.5f;
		float beta = 6.0f;
	};

	struct TalCandidate
	{
		int truth_idx = -1;
		int point_idx = -1;
		float class_score = 0.0f;
		float overlap = 0.0f;
		bool anchor_inside_truth = true;
	};

	struct TalAssignment
	{
		int truth_idx = -1;
		float metric = 0.0f;
		float overlap = 0.0f;
		float target_score = 0.0f;
	};

	struct BranchLossScaleConfig
	{
		int branch_count = 0;
		int inference_branch = 0;
		float aux_loss_weight = 0.25f;
		float batch_size = 1.0f;
		float target_scores_sum = 1.0f;
		float cls_normalizer = 0.5f;
		float box_normalizer = 7.5f;
		float dfl_normalizer = 1.5f;
	};

	struct BranchLossScales
	{
		float branch_weight = 0.0f;
		float cls_scale = 0.0f;
		float box_scale = 0.0f;
		float dfl_scale = 0.0f;
	};

	struct YoloV9InputShape
	{
		int width = 0;
		int height = 0;
		int channels = 0;
		int stride = 0;
	};

	struct YoloV9ResizeShapeConfig
	{
		int net_width = 0;
		int net_height = 0;
		int classes = 0;
		int reg_max = 0;
		int branch_count = 0;
		int inference_branch = 0;
		std::vector<YoloV9InputShape> inputs;
	};

	struct YoloV9ResizeShapeSummary
	{
		int expected_channels = 0;
		std::vector<int> input_sizes;
		int aggregate_inputs = 0;
		int inference_outputs = 0;
	};

	std::vector<float> dfl_project_one_anchor(const std::vector<float> & logits, int reg_max);
	AnchorGrid make_anchor_points(const std::vector<FeatureMapShape> & levels, float grid_cell_offset = 0.5f);
	BoxXYXY dist2bbox_xyxy(const Point & anchor, const std::array<float, 4> & ltrb);
	BoxXYWH dist2bbox_xywh(const Point & anchor, const std::array<float, 4> & ltrb);
	std::vector<NoObjectnessDetection> extract_no_objectness_detections(
		const std::vector<DecodedPrediction> & predictions,
		float threshold);
	DDetectShapeSummary summarize_ddetect_shapes(const DDetectShapeConfig & config);
	std::vector<TalAssignment> assign_tal_targets(
		const TalAssignerConfig & config,
		const std::vector<TalCandidate> & candidates);
	float normalized_target_scores_sum(const std::vector<TalAssignment> & assignments);
	BranchLossScales branch_loss_scales(
		const BranchLossScaleConfig & config,
		int branch,
		float target_score);
	YoloV9ResizeShapeSummary summarize_yolov9_resize_shapes(const YoloV9ResizeShapeConfig & config);
}
