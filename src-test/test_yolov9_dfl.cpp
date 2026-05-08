#include <gtest/gtest.h>

#include "yolov9_test_helpers.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "darknet_internal.hpp"

namespace
{
	std::vector<float> logits_from_probabilities(const std::vector<std::array<float, 4>> & probabilities)
	{
		std::vector<float> logits;
		logits.reserve(probabilities.size() * 4);
		for (const auto & side : probabilities)
		{
			for (const float probability : side)
			{
				logits.push_back(std::log(probability));
			}
		}
		return logits;
	}

	std::string find_repo_file(const std::string & relative_path)
	{
		const std::vector<std::string> candidates =
		{
			relative_path,
			"../" + relative_path,
			"../../" + relative_path
		};

		for (const auto & candidate : candidates)
		{
			std::ifstream stream(candidate);
			if (stream.good())
			{
				return candidate;
			}
		}

		throw std::runtime_error("unable to locate " + relative_path);
	}
}

TEST(YOLOv9DFL, UniformLogitsProjectToBinMean)
{
	const std::vector<float> logits(4 * 4, 0.0f);

	const auto distances = YoloV9Test::dfl_project_one_anchor(logits, 4);

	ASSERT_EQ(4, distances.size());
	for (const float distance : distances)
	{
		ASSERT_FLOAT_EQ(1.5f, distance);
	}
}

TEST(YOLOv9LayerRuntime, DFLProjectionMatchesReferenceHelper)
{
	const auto logits = logits_from_probabilities(
		{
			std::array<float, 4>{0.10f, 0.20f, 0.30f, 0.40f}
		});

	ASSERT_NEAR(2.0f, yolov9_dfl_project(logits.data(), 4), 1.0e-6f);
}

TEST(YOLOv9LayerRuntime, DFLCrossEntropyReturnsReferenceDeltas)
{
	const auto logits = logits_from_probabilities(
		{
			std::array<float, 4>{0.10f, 0.20f, 0.30f, 0.40f}
		});
	float deltas[4] = {};

	const float loss = yolov9_dfl_cross_entropy_delta(logits.data(), 4, 1.25f, 2.0f, deltas);
	const float expected_loss = 2.0f * (0.75f * -std::log(0.20f) + 0.25f * -std::log(0.30f));

	ASSERT_NEAR(expected_loss, loss, 1.0e-6f);
	ASSERT_NEAR(-0.20f, deltas[0], 1.0e-6f);
	ASSERT_NEAR(1.10f, deltas[1], 1.0e-6f);
	ASSERT_NEAR(-0.10f, deltas[2], 1.0e-6f);
	ASSERT_NEAR(-0.80f, deltas[3], 1.0e-6f);
}

TEST(YOLOv9LayerRuntime, Dist2BBoxProducesNormalizedXYWH)
{
	const float ltrb[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	const Darknet::Box box = yolov9_dist2bbox(3.5f, 4.5f, ltrb, 8.0f, 80, 160);

	ASSERT_FLOAT_EQ(0.45f, box.x);
	ASSERT_FLOAT_EQ(0.275f, box.y);
	ASSERT_FLOAT_EQ(0.4f, box.w);
	ASSERT_FLOAT_EQ(0.3f, box.h);
}

TEST(YOLOv9DFL, LogProbabilityLogitsMatchExpectedProjection)
{
	const auto logits = logits_from_probabilities(
		{
			std::array<float, 4>{0.10f, 0.20f, 0.30f, 0.40f},
			std::array<float, 4>{0.70f, 0.10f, 0.10f, 0.10f},
			std::array<float, 4>{0.25f, 0.25f, 0.25f, 0.25f},
			std::array<float, 4>{0.40f, 0.30f, 0.20f, 0.10f}
		});

	const auto distances = YoloV9Test::dfl_project_one_anchor(logits, 4);

	ASSERT_NEAR(2.0f, distances[0], 1.0e-6f);
	ASSERT_NEAR(0.6f, distances[1], 1.0e-6f);
	ASSERT_NEAR(1.5f, distances[2], 1.0e-6f);
	ASSERT_NEAR(1.0f, distances[3], 1.0e-6f);
}

TEST(YOLOv9DFL, RejectsInvalidLogitShape)
{
	ASSERT_THROW(YoloV9Test::dfl_project_one_anchor(std::vector<float>(15, 0.0f), 4), std::invalid_argument);
	ASSERT_THROW(YoloV9Test::dfl_project_one_anchor(std::vector<float>(16, 0.0f), 0), std::invalid_argument);
}

TEST(YOLOv9Anchors, GeneratesReferenceGridCentersAndStrides)
{
	const auto grid = YoloV9Test::make_anchor_points(
		{
			YoloV9Test::FeatureMapShape{2, 2, 8.0f},
			YoloV9Test::FeatureMapShape{1, 3, 16.0f}
		});

	ASSERT_EQ(7, grid.points.size());
	ASSERT_EQ(7, grid.strides.size());

	const std::vector<YoloV9Test::Point> expected_points =
	{
		{0.5f, 0.5f}, {1.5f, 0.5f}, {0.5f, 1.5f}, {1.5f, 1.5f},
		{0.5f, 0.5f}, {0.5f, 1.5f}, {0.5f, 2.5f}
	};
	const std::vector<float> expected_strides = {8.0f, 8.0f, 8.0f, 8.0f, 16.0f, 16.0f, 16.0f};

	for (std::size_t idx = 0; idx < expected_points.size(); ++idx)
	{
		ASSERT_FLOAT_EQ(expected_points[idx].x, grid.points[idx].x);
		ASSERT_FLOAT_EQ(expected_points[idx].y, grid.points[idx].y);
		ASSERT_FLOAT_EQ(expected_strides[idx], grid.strides[idx]);
	}
}

TEST(YOLOv9Anchors, RejectsInvalidFeatureLevels)
{
	ASSERT_THROW(YoloV9Test::make_anchor_points({}), std::invalid_argument);
	ASSERT_THROW(YoloV9Test::make_anchor_points({YoloV9Test::FeatureMapShape{0, 2, 8.0f}}), std::invalid_argument);
	ASSERT_THROW(YoloV9Test::make_anchor_points({YoloV9Test::FeatureMapShape{2, 2, 0.0f}}), std::invalid_argument);
}

TEST(YOLOv9Dist2BBox, DecodesLTRBToXYXYAndXYWH)
{
	const YoloV9Test::Point anchor{3.5f, 4.5f};
	const std::array<float, 4> ltrb{1.0f, 2.0f, 3.0f, 4.0f};

	const auto xyxy = YoloV9Test::dist2bbox_xyxy(anchor, ltrb);
	ASSERT_FLOAT_EQ(2.5f, xyxy.x1);
	ASSERT_FLOAT_EQ(2.5f, xyxy.y1);
	ASSERT_FLOAT_EQ(6.5f, xyxy.x2);
	ASSERT_FLOAT_EQ(8.5f, xyxy.y2);

	const auto xywh = YoloV9Test::dist2bbox_xywh(anchor, ltrb);
	ASSERT_FLOAT_EQ(4.5f, xywh.x);
	ASSERT_FLOAT_EQ(5.5f, xywh.y);
	ASSERT_FLOAT_EQ(4.0f, xywh.w);
	ASSERT_FLOAT_EQ(6.0f, xywh.h);
}

TEST(YOLOv9NoObjectness, UsesClassScoresDirectly)
{
	const std::vector<YoloV9Test::DecodedPrediction> predictions =
	{
		{YoloV9Test::BoxXYWH{10.0f, 20.0f, 4.0f, 6.0f}, {0.20f, 0.80f, 0.49f}},
		{YoloV9Test::BoxXYWH{30.0f, 40.0f, 8.0f, 10.0f}, {0.50f, 0.10f, 0.90f}},
		{YoloV9Test::BoxXYWH{50.0f, 60.0f, 12.0f, 14.0f}, {0.10f, 0.20f, 0.30f}}
	};

	const auto detections = YoloV9Test::extract_no_objectness_detections(predictions, 0.50f);

	ASSERT_EQ(2, detections.size());

	ASSERT_FLOAT_EQ(1.0f, detections[0].objectness);
	ASSERT_FLOAT_EQ(10.0f, detections[0].box.x);
	ASSERT_FLOAT_EQ(0.0f, detections[0].probabilities[0]);
	ASSERT_FLOAT_EQ(0.80f, detections[0].probabilities[1]);
	ASSERT_FLOAT_EQ(0.0f, detections[0].probabilities[2]);

	ASSERT_FLOAT_EQ(1.0f, detections[1].objectness);
	ASSERT_FLOAT_EQ(30.0f, detections[1].box.x);
	ASSERT_FLOAT_EQ(0.50f, detections[1].probabilities[0]);
	ASSERT_FLOAT_EQ(0.0f, detections[1].probabilities[1]);
	ASSERT_FLOAT_EQ(0.90f, detections[1].probabilities[2]);
}

TEST(YOLOv9NoObjectness, RejectsMalformedPredictions)
{
	ASSERT_THROW(
		YoloV9Test::extract_no_objectness_detections(
			{{YoloV9Test::BoxXYWH{0.0f, 0.0f, 1.0f, 1.0f}, {}}},
			0.25f),
		std::invalid_argument);
	ASSERT_THROW(YoloV9Test::extract_no_objectness_detections({}, -0.01f), std::invalid_argument);
	ASSERT_THROW(YoloV9Test::extract_no_objectness_detections({}, 1.01f), std::invalid_argument);
}

TEST(YOLOv9ShapeScaffold, SummarizesDDetectTrainingAndInferenceShapes)
{
	const YoloV9Test::DDetectShapeConfig config =
	{
		2,
		3,
		16,
		{
			YoloV9Test::FeatureMapShape{80, 80, 8.0f},
			YoloV9Test::FeatureMapShape{40, 40, 16.0f},
			YoloV9Test::FeatureMapShape{20, 20, 32.0f}
		}
	};

	const auto summary = YoloV9Test::summarize_ddetect_shapes(config);

	ASSERT_EQ(67, summary.logits_channels_per_point);
	ASSERT_EQ(7, summary.decoded_channels_per_point);
	ASSERT_EQ(8400, summary.total_points);

	ASSERT_EQ(3, summary.points_per_level.size());
	ASSERT_EQ(6400, summary.points_per_level[0]);
	ASSERT_EQ(1600, summary.points_per_level[1]);
	ASSERT_EQ(400, summary.points_per_level[2]);

	ASSERT_EQ((std::array<int, 4>{2, 67, 80, 80}), summary.training_tensor_shapes[0]);
	ASSERT_EQ((std::array<int, 4>{2, 67, 40, 40}), summary.training_tensor_shapes[1]);
	ASSERT_EQ((std::array<int, 4>{2, 67, 20, 20}), summary.training_tensor_shapes[2]);
	ASSERT_EQ((std::array<int, 3>{2, 7, 8400}), summary.inference_tensor_shape);

	ASSERT_NE(config.classes + 5, summary.decoded_channels_per_point);
}

TEST(YOLOv9ShapeScaffold, RejectsInvalidDDetectShapes)
{
	ASSERT_THROW(YoloV9Test::summarize_ddetect_shapes(YoloV9Test::DDetectShapeConfig{}), std::invalid_argument);
	ASSERT_THROW(
		YoloV9Test::summarize_ddetect_shapes(
			YoloV9Test::DDetectShapeConfig{1, 80, 16, {YoloV9Test::FeatureMapShape{0, 1, 8.0f}}}),
		std::invalid_argument);
}

TEST(YOLOv9Cfg, ParsesGelanTinyLegoGears)
{
	const std::string cfg = find_repo_file("cfg/gelan-t-legogears.cfg");
	Darknet::Network net = parse_network_cfg_custom(cfg.c_str(), 1, 1);

	ASSERT_GT(net.n, 0);
	const Darknet::Layer & output = net.layers[net.n - 1];
	ASSERT_EQ(Darknet::ELayerType::YOLOV9, output.type);
	ASSERT_EQ(5, output.classes);
	ASSERT_EQ(16, output.reg_max);
	ASSERT_EQ(3, output.n);
	ASSERT_EQ(8, output.strides[0]);
	ASSERT_EQ(16, output.strides[1]);
	ASSERT_EQ(32, output.strides[2]);

	for (int idx = 0; idx < output.n; ++idx)
	{
		const Darknet::Layer & prediction = net.layers[output.input_layers[idx]];
		ASSERT_EQ(69, prediction.out_c);
	}

	free_network(net);
}

TEST(YOLOv9Cfg, ParsesDualBranchTinyLegoGears)
{
	const std::string cfg = find_repo_file("cfg/yolov9-t-legogears.cfg");
	Darknet::Network net = parse_network_cfg_custom(cfg.c_str(), 1, 1);

	ASSERT_GT(net.n, 0);
	const Darknet::Layer & output = net.layers[net.n - 1];
	ASSERT_EQ(Darknet::ELayerType::YOLOV9, output.type);
	ASSERT_EQ(5, output.classes);
	ASSERT_EQ(16, output.reg_max);
	ASSERT_EQ(3, output.n);
	ASSERT_EQ(6, output.total);
	ASSERT_EQ(2, output.branch_count);
	ASSERT_EQ(1, output.inference_branch);
	ASSERT_FLOAT_EQ(0.25f, output.aux_loss_weight);
	ASSERT_EQ(8, output.strides[0]);
	ASSERT_EQ(16, output.strides[1]);
	ASSERT_EQ(32, output.strides[2]);

	for (int idx = 0; idx < output.total; ++idx)
	{
		const Darknet::Layer & prediction = net.layers[output.input_layers[idx]];
		ASSERT_EQ(69, prediction.out_c);
	}

	free_network(net);
}
