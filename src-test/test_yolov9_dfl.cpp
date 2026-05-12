#include <gtest/gtest.h>

#include "yolov9_test_helpers.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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

	std::filesystem::path write_temp_cfg(const std::string & filename, const std::string & contents)
	{
		const auto path = std::filesystem::temp_directory_path() / filename;
		std::ofstream stream(path);
		if (not stream.good())
		{
			throw std::runtime_error("unable to write " + path.string());
		}
		stream << contents;
		return path;
	}

	std::string minimal_yolov9_cfg(const std::string & iou_loss)
	{
		return
			"[net]\n"
			"batch=1\n"
			"subdivisions=1\n"
			"width=64\n"
			"height=64\n"
			"channels=3\n"
			"max_batches=1\n"
			"learning_rate=0.001\n"
			"\n"
			"[convolutional]\n"
			"filters=69\n"
			"size=1\n"
			"stride=8\n"
			"pad=0\n"
			"activation=linear\n"
			"\n"
			"[yolov9]\n"
			"classes=5\n"
			"reg_max=16\n"
			"layers=-1\n"
			"strides=8\n"
			"iou_loss=" + iou_loss + "\n";
	}

	float synthetic_loss_logit(const int channel, const int x, const int y, const int reg_max)
	{
		if (channel < 4 * reg_max)
		{
			const int side = channel / reg_max;
			const int bin = channel % reg_max;
			return -0.35f + 0.19f * bin + 0.07f * side + 0.025f * x - 0.015f * y;
		}

		const int class_id = channel - 4 * reg_max;
		if (class_id == 0)
		{
			return -1.2f + 0.1f * x;
		}
		if (class_id == 1)
		{
			return -0.2f + 0.25f * x + 0.15f * y;
		}
		return -0.5f + 0.1f * y;
	}

	int chw_index(const int channel, const int y, const int x, const int height, const int width)
	{
		return channel * height * width + y * width + x;
	}

	class EnvVarGuard
	{
	public:
		explicit EnvVarGuard(const char * env_name) :
			name(env_name)
		{
			const char * value = std::getenv(env_name);
			if (value != nullptr)
			{
				had_value = true;
				saved_value = value;
			}
		}

		~EnvVarGuard()
		{
			if (had_value)
			{
				setenv(name.c_str(), saved_value.c_str(), 1);
			}
			else
			{
				unsetenv(name.c_str());
			}
		}

		EnvVarGuard(const EnvVarGuard &) = delete;
		EnvVarGuard & operator=(const EnvVarGuard &) = delete;

	private:
		std::string name;
		std::string saved_value;
		bool had_value = false;
	};
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

TEST(YOLOv9TAL, TopKAssignmentUsesAlignmentMetric)
{
	const auto assignments = YoloV9Test::assign_tal_targets(
		YoloV9Test::TalAssignerConfig{4, 1, 2, 1.0f, 1.0f},
		{
			YoloV9Test::TalCandidate{0, 0, 0.90f, 0.20f, true},
			YoloV9Test::TalCandidate{0, 1, 0.50f, 0.80f, true},
			YoloV9Test::TalCandidate{0, 2, 0.70f, 0.70f, true},
			YoloV9Test::TalCandidate{0, 3, 0.60f, 0.10f, true}
		});

	ASSERT_EQ(4, assignments.size());
	ASSERT_EQ(-1, assignments[0].truth_idx);
	ASSERT_EQ(0, assignments[1].truth_idx);
	ASSERT_EQ(0, assignments[2].truth_idx);
	ASSERT_EQ(-1, assignments[3].truth_idx);
	ASSERT_NEAR(0.4f, assignments[1].metric, 1.0e-6f);
	ASSERT_NEAR(0.49f, assignments[2].metric, 1.0e-6f);
	ASSERT_NEAR(0.8f * 0.4f / 0.49f, assignments[1].target_score, 1.0e-6f);
	ASSERT_NEAR(0.8f, assignments[2].target_score, 1.0e-6f);
}

TEST(YOLOv9TAL, OverlapConflictResolutionKeepsHighestOverlapTruth)
{
	const auto assignments = YoloV9Test::assign_tal_targets(
		YoloV9Test::TalAssignerConfig{1, 2, 1, 1.0f, 1.0f},
		{
			YoloV9Test::TalCandidate{0, 0, 0.90f, 0.40f, true},
			YoloV9Test::TalCandidate{1, 0, 0.20f, 0.80f, true}
		});

	ASSERT_EQ(1, assignments.size());
	ASSERT_EQ(1, assignments[0].truth_idx);
	ASSERT_NEAR(0.80f, assignments[0].overlap, 1.0e-6f);
	ASSERT_NEAR(0.80f, assignments[0].target_score, 1.0e-6f);
}

TEST(YOLOv9TAL, NormalizesTargetScoresByBestMetricAndOverlap)
{
	const auto assignments = YoloV9Test::assign_tal_targets(
		YoloV9Test::TalAssignerConfig{3, 1, 3, 1.0f, 1.0f},
		{
			YoloV9Test::TalCandidate{0, 0, 0.40f, 0.50f, true},
			YoloV9Test::TalCandidate{0, 1, 0.50f, 0.80f, true},
			YoloV9Test::TalCandidate{0, 2, 0.50f, 0.20f, true}
		});

	ASSERT_NEAR(0.40f, assignments[0].target_score, 1.0e-6f);
	ASSERT_NEAR(0.80f, assignments[1].target_score, 1.0e-6f);
	ASSERT_NEAR(0.20f, assignments[2].target_score, 1.0e-6f);
	ASSERT_NEAR(1.40f, YoloV9Test::normalized_target_scores_sum(assignments), 1.0e-6f);
	ASSERT_FLOAT_EQ(1.0f, YoloV9Test::normalized_target_scores_sum({}));
}

TEST(YOLOv9LossScales, AppliesAuxiliaryAndMainBranchWeights)
{
	const YoloV9Test::BranchLossScaleConfig config{2, 1, 0.25f, 2.0f, 4.0f, 0.5f, 7.5f, 1.5f};

	const auto aux = YoloV9Test::branch_loss_scales(config, 0, 0.8f);
	const auto main = YoloV9Test::branch_loss_scales(config, 1, 0.8f);

	ASSERT_FLOAT_EQ(0.25f, aux.branch_weight);
	ASSERT_FLOAT_EQ(1.0f, main.branch_weight);
	ASSERT_NEAR(main.cls_scale * 0.25f, aux.cls_scale, 1.0e-6f);
	ASSERT_NEAR(main.box_scale * 0.25f, aux.box_scale, 1.0e-6f);
	ASSERT_NEAR(main.dfl_scale * 0.25f, aux.dfl_scale, 1.0e-6f);
	ASSERT_NEAR(0.25f, main.cls_scale, 1.0e-6f);
	ASSERT_NEAR(3.0f, main.box_scale, 1.0e-6f);
	ASSERT_NEAR(0.15f, main.dfl_scale, 1.0e-6f);
}

TEST(YOLOv9LayerRuntime, TrainingLossMatchesPyTorchSyntheticFixture)
{
	constexpr int batch = 1;
	constexpr int classes = 3;
	constexpr int reg_max = 4;
	constexpr int feature_w = 4;
	constexpr int feature_h = 4;
	constexpr int stride = 8;
	constexpr int net_w = feature_w * stride;
	constexpr int net_h = feature_h * stride;
	constexpr int channels = classes + 4 * reg_max;
	constexpr int prediction_outputs = channels * feature_w * feature_h;
	constexpr int max_boxes = 32;

	std::vector<float> prediction_output(prediction_outputs, 0.0f);
	std::vector<float> prediction_delta(prediction_outputs, 0.0f);
	for (int y = 0; y < feature_h; ++y)
	{
		for (int x = 0; x < feature_w; ++x)
		{
			for (int channel = 0; channel < channels; ++channel)
			{
				prediction_output[chw_index(channel, y, x, feature_h, feature_w)] = synthetic_loss_logit(channel, x, y, reg_max);
			}
		}
	}

	std::vector<int> input_layers{0};
	std::vector<int> input_sizes{prediction_outputs};
	std::vector<int> strides{stride};
	std::vector<float> yolov9_output(prediction_outputs, 0.0f);
	std::vector<float> yolov9_delta(prediction_outputs, 0.0f);
	std::vector<float> yolov9_cost(1, 0.0f);

	std::array<Darknet::Layer, 2> layers{};
	Darknet::Layer & prediction = layers[0];
	prediction.batch = batch;
	prediction.outputs = prediction_outputs;
	prediction.out_w = feature_w;
	prediction.out_h = feature_h;
	prediction.out_c = channels;
	prediction.output = prediction_output.data();
	prediction.delta = prediction_delta.data();

	Darknet::Layer & yolo = layers[1];
	yolo.type = Darknet::ELayerType::YOLOV9;
	yolo.batch = batch;
	yolo.n = 1;
	yolo.total = 1;
	yolo.input_layers = input_layers.data();
	yolo.input_sizes = input_sizes.data();
	yolo.strides = strides.data();
	yolo.classes = classes;
	yolo.reg_max = reg_max;
	yolo.max_boxes = max_boxes;
	yolo.truth_size = 6;
	yolo.truths = max_boxes * yolo.truth_size;
	yolo.branch_count = 1;
	yolo.inference_branch = 0;
	yolo.aux_loss_weight = 0.25f;
	yolo.box_normalizer = 7.5f;
	yolo.cls_normalizer = 0.5f;
	yolo.dfl_normalizer = 1.5f;
	yolo.tal_topk = 10;
	yolo.tal_alpha = 0.5f;
	yolo.tal_beta = 6.0f;
	yolo.iou_loss = CIOU;
	yolo.outputs = prediction_outputs;
	yolo.output = yolov9_output.data();
	yolo.delta = yolov9_delta.data();
	yolo.cost = yolov9_cost.data();

	Darknet::Network net{};
	net.n = static_cast<int>(layers.size());
	net.batch = batch;
	net.w = net_w;
	net.h = net_h;
	net.c = 3;
	net.layers = layers.data();
	net.loss_scale = 1.0f;

	std::vector<float> truth(static_cast<std::size_t>(batch) * yolo.truths, 0.0f);
	truth[0] = 0.5f;
	truth[1] = 0.5f;
	truth[2] = 0.75f;
	truth[3] = 0.75f;
	truth[4] = 1.0f;

	Darknet::NetworkState state{};
	state.net = net;
	state.truth = truth.data();
	state.train = 1;

	forward_yolov9_layer(yolo, state);

	ASSERT_NEAR(12.270484f, yolo.cost[0], 5.0e-3f);
	ASSERT_FLOAT_EQ(0.0f, prediction_delta[chw_index(0, 0, 0, feature_h, feature_w)]);
	ASSERT_NEAR(-0.0563491f, prediction_delta[chw_index(4 * reg_max + 0, 0, 0, feature_h, feature_w)], 1.0e-5f);
	ASSERT_NEAR(-0.1095860f, prediction_delta[chw_index(4 * reg_max + 1, 0, 0, feature_h, feature_w)], 1.0e-5f);
	ASSERT_NEAR(-0.0919065f, prediction_delta[chw_index(4 * reg_max + 2, 0, 0, feature_h, feature_w)], 1.0e-5f);
	ASSERT_NEAR(-0.0139559f, prediction_delta[chw_index(4 * reg_max + 1, 1, 1, feature_h, feature_w)], 1.0e-5f);
	ASSERT_NEAR(0.0719648f, prediction_delta[chw_index(0, 1, 1, feature_h, feature_w)], 1.0e-3f);
}

TEST(YOLOv9ShapeScaffold, ValidatesResizeShapeExpectations)
{
	const YoloV9Test::YoloV9ResizeShapeConfig config =
	{
		640,
		640,
		5,
		16,
		2,
		1,
		{
			YoloV9Test::YoloV9InputShape{80, 80, 69, 8},
			YoloV9Test::YoloV9InputShape{40, 40, 69, 16},
			YoloV9Test::YoloV9InputShape{20, 20, 69, 32},
			YoloV9Test::YoloV9InputShape{80, 80, 69, 8},
			YoloV9Test::YoloV9InputShape{40, 40, 69, 16},
			YoloV9Test::YoloV9InputShape{20, 20, 69, 32}
		}
	};

	const auto summary = YoloV9Test::summarize_yolov9_resize_shapes(config);

	ASSERT_EQ(69, summary.expected_channels);
	ASSERT_EQ(6, summary.input_sizes.size());
	ASSERT_EQ(579600, summary.inference_outputs);
	ASSERT_EQ(1159200, summary.aggregate_inputs);
	ASSERT_THROW(
		YoloV9Test::summarize_yolov9_resize_shapes(
			YoloV9Test::YoloV9ResizeShapeConfig{640, 640, 5, 16, 1, 0, {YoloV9Test::YoloV9InputShape{80, 80, 70, 8}}}),
		std::invalid_argument);
	ASSERT_THROW(
		YoloV9Test::summarize_yolov9_resize_shapes(
			YoloV9Test::YoloV9ResizeShapeConfig{640, 640, 5, 16, 1, 0, {YoloV9Test::YoloV9InputShape{80, 80, 69, 16}}}),
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

TEST(YOLOv9InferenceOptimizer, FusesGelanTinyRepConvNInMemory)
{
	const EnvVarGuard optimizer_env("DARKNET_DISABLE_INFERENCE_OPTIMIZER");
	unsetenv("DARKNET_DISABLE_INFERENCE_OPTIMIZER");
	const std::string cfg = find_repo_file("cfg/gelan-t-legogears.cfg");
	Darknet::Network net = parse_network_cfg_custom(cfg.c_str(), 1, 1);

	fuse_conv_batchnorm(net);
	optimize_network_for_inference(net);

	ASSERT_EQ(1, net.inference_optimized);
	ASSERT_EQ(42, net.inference_optimizer_fused_repconvn);
	ASSERT_GE(net.inference_optimizer_skipped_layers, net.inference_optimizer_fused_repconvn * 3);

	ASSERT_EQ(Darknet::ELayerType::CONVOLUTIONAL, net.layers[15].type);
	ASSERT_EQ(SWISH, net.layers[15].activation);
	ASSERT_EQ(1, net.layers[16].inference_skip);
	ASSERT_EQ(14, resolve_inference_layer_index(net, 16));
	ASSERT_EQ(1, net.layers[17].inference_skip);
	ASSERT_EQ(15, resolve_inference_layer_index(net, 17));
	ASSERT_EQ(1, net.layers[18].inference_skip);
	ASSERT_EQ(15, resolve_inference_layer_index(net, 18));

	free_network(net);
}

TEST(YOLOv9InferenceOptimizer, DisableEnvKeepsTrainingGraphUnmodified)
{
	const EnvVarGuard optimizer_env("DARKNET_DISABLE_INFERENCE_OPTIMIZER");
	setenv("DARKNET_DISABLE_INFERENCE_OPTIMIZER", "1", 1);
	const std::string cfg = find_repo_file("cfg/gelan-t-legogears.cfg");
	Darknet::Network net = parse_network_cfg_custom(cfg.c_str(), 1, 1);

	fuse_conv_batchnorm(net);
	optimize_network_for_inference(net);

	ASSERT_EQ(0, net.inference_optimized);
	ASSERT_EQ(0, net.inference_optimizer_fused_repconvn);
	ASSERT_EQ(0, net.inference_optimizer_aliased_routes);
	ASSERT_EQ(0, net.inference_optimizer_pruned_layers);
	ASSERT_EQ(0, net.inference_optimizer_skipped_layers);
	ASSERT_EQ(0, net.layers[16].inference_skip);
	ASSERT_EQ(16, resolve_inference_layer_index(net, 16));
	ASSERT_EQ(0, net.layers[17].inference_skip);
	ASSERT_EQ(17, resolve_inference_layer_index(net, 17));
	ASSERT_EQ(0, net.layers[18].inference_skip);
	ASSERT_EQ(18, resolve_inference_layer_index(net, 18));

	free_network(net);
}

TEST(YOLOv9InferenceOptimizer, PrunesDualBranchAuxiliaryPathForInference)
{
	const EnvVarGuard optimizer_env("DARKNET_DISABLE_INFERENCE_OPTIMIZER");
	unsetenv("DARKNET_DISABLE_INFERENCE_OPTIMIZER");
	const std::string cfg = find_repo_file("cfg/yolov9-t-legogears.cfg");
	Darknet::Network net = parse_network_cfg_custom(cfg.c_str(), 1, 1);

	fuse_conv_batchnorm(net);
	optimize_network_for_inference(net);

	ASSERT_EQ(1, net.inference_optimized);
	ASSERT_EQ(93, net.inference_optimizer_pruned_layers);

	const Darknet::Layer & output = net.layers[net.n - 1];
	ASSERT_EQ(Darknet::ELayerType::YOLOV9, output.type);
	ASSERT_EQ(2, output.branch_count);
	ASSERT_EQ(1, output.inference_branch);

	const std::array<int, 3> auxiliary_outputs = {527, 536, 545};
	for (const int layer_index : auxiliary_outputs)
	{
		ASSERT_EQ(1, net.layers[layer_index].inference_skip) << "layer " << layer_index;
		const int resolved_index = resolve_inference_layer_index(net, layer_index);
		ASSERT_GE(resolved_index, 0);
		ASSERT_LT(resolved_index, net.n);
		ASSERT_FALSE(net.layers[resolved_index].inference_skip) << "layer " << layer_index;
	}

	const std::array<int, 3> main_outputs = {554, 563, 572};
	for (int scale = 0; scale < output.n; ++scale)
	{
		const int slot = output.inference_branch * output.n + scale;
		ASSERT_EQ(main_outputs[scale], output.input_layers[slot]);
		const int input_index = resolve_inference_layer_index(net, output.input_layers[slot]);
		ASSERT_GE(input_index, 0);
		ASSERT_LT(input_index, net.n);
		ASSERT_FALSE(net.layers[input_index].inference_skip);
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
	ASSERT_EQ(CIOU, output.iou_loss);
	ASSERT_EQ(8, output.strides[0]);
	ASSERT_EQ(16, output.strides[1]);
	ASSERT_EQ(32, output.strides[2]);

	int main_branch_outputs = 0;
	for (int idx = 0; idx < output.total; ++idx)
	{
		const Darknet::Layer & prediction = net.layers[output.input_layers[idx]];
		ASSERT_EQ(69, prediction.out_c);
		const int scale = idx % output.n;
		ASSERT_EQ(net.w, prediction.out_w * output.strides[scale]);
		ASSERT_EQ(net.h, prediction.out_h * output.strides[scale]);
		if (idx >= output.n)
		{
			main_branch_outputs += prediction.outputs;
		}
	}
	ASSERT_EQ(main_branch_outputs, output.outputs);

	free_network(net);
}

TEST(YOLOv9Cfg, RejectsNonCiouIouLoss)
{
	const auto cfg = write_temp_cfg("darknet_yolov9_non_ciou_test.cfg", minimal_yolov9_cfg("giou"));

	EXPECT_DEATH(
		{
			std::ofstream devnull("/dev/null");
			Darknet::CfgAndState::get().output = &devnull;
			Darknet::Network net = parse_network_cfg_custom(cfg.string().c_str(), 1, 1);
			free_network(net);
		},
		".*");

	std::filesystem::remove(cfg);
}

TEST(YOLOv9LayerRuntime, ResizeKeepsDualBranchShapeConsistent)
{
	const std::string cfg = find_repo_file("cfg/yolov9-t-legogears.cfg");
	Darknet::Network net = parse_network_cfg_custom(cfg.c_str(), 1, 1);

	resize_network(&net, 320, 320);

	const Darknet::Layer & output = net.layers[net.n - 1];
	ASSERT_EQ(Darknet::ELayerType::YOLOV9, output.type);
	ASSERT_EQ(320, net.w);
	ASSERT_EQ(320, net.h);
	ASSERT_EQ(69, output.classes + 4 * output.reg_max);

	int selected_outputs = 0;
	for (int branch = 0; branch < output.branch_count; ++branch)
	{
		for (int scale = 0; scale < output.n; ++scale)
		{
			const int slot = branch * output.n + scale;
			const Darknet::Layer & prediction = net.layers[output.input_layers[slot]];
			ASSERT_GT(prediction.out_w, 0);
			ASSERT_GT(prediction.out_h, 0);
			ASSERT_EQ(69, prediction.out_c);
			ASSERT_EQ(net.w, prediction.out_w * output.strides[scale]);
			ASSERT_EQ(net.h, prediction.out_h * output.strides[scale]);
			ASSERT_EQ(prediction.out_w * prediction.out_h * prediction.out_c, prediction.outputs);
			if (branch == output.inference_branch)
			{
				selected_outputs += prediction.outputs;
			}
		}
	}
	ASSERT_EQ(selected_outputs, output.outputs);

	free_network(net);
}
