#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "darknet_internal.hpp"

namespace
{
	void set_iteration(Darknet::Network & net, const int iteration)
	{
		*net.seen = static_cast<uint64_t>(iteration) * static_cast<uint64_t>(net.batch * net.subdivisions);
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

	class GpuIndexGuard
	{
	public:
		explicit GpuIndexGuard(const int temporary_gpu_index)
		{
			auto & cfg_and_state = Darknet::CfgAndState::get();
			saved_gpu_index = cfg_and_state.gpu_index;
			cfg_and_state.gpu_index = temporary_gpu_index;
		}

		~GpuIndexGuard()
		{
			Darknet::CfgAndState::get().gpu_index = saved_gpu_index;
		}

		GpuIndexGuard(const GpuIndexGuard &) = delete;
		GpuIndexGuard & operator=(const GpuIndexGuard &) = delete;

	private:
		int saved_gpu_index = -1;
	};
}

TEST(DarknetScheduler, LinearFinalInterpolatesToConfiguredFinalRate)
{
	Darknet::Network net = make_network(0);
	net.batch = 1;
	net.subdivisions = 1;
	net.policy = LINEAR_FINAL;
	net.learning_rate = 0.01f;
	net.final_learning_rate = 0.0001f;
	net.max_batches = 1000;

	set_iteration(net, 0);
	ASSERT_NEAR(0.01f, get_current_rate(net), 1.0e-8f);

	set_iteration(net, 500);
	ASSERT_NEAR(0.00505f, get_current_rate(net), 1.0e-8f);

	set_iteration(net, 1000);
	ASSERT_NEAR(0.0001f, get_current_rate(net), 1.0e-8f);

	free_network(net);
}

TEST(DarknetScheduler, WarmupSeparatesNonBiasBiasAndMomentum)
{
	Darknet::Network net = make_network(0);
	net.batch = 1;
	net.subdivisions = 1;
	net.policy = LINEAR_FINAL;
	net.learning_rate = 0.01f;
	net.final_learning_rate = 0.0001f;
	net.max_batches = 1000;
	net.momentum = 0.937f;
	net.warmup_iterations = 100;
	net.warmup_nonbias_lr_start = 0.0f;
	net.warmup_bias_lr = 0.1f;
	net.warmup_momentum = 0.8f;

	set_iteration(net, 0);
	ASSERT_FLOAT_EQ(0.0f, get_current_rate(net));
	ASSERT_FLOAT_EQ(0.1f, get_current_bias_rate(net));
	ASSERT_FLOAT_EQ(0.8f, get_current_momentum(net));

	set_iteration(net, 50);
	const float target_mid = 0.01f + (0.0001f - 0.01f) * 0.05f;
	ASSERT_NEAR(target_mid * 0.5f, get_current_rate(net), 1.0e-7f);
	ASSERT_NEAR(0.1f + (target_mid - 0.1f) * 0.5f, get_current_bias_rate(net), 1.0e-7f);
	ASSERT_NEAR(0.8685f, get_current_momentum(net), 1.0e-6f);

	set_iteration(net, 100);
	ASSERT_NEAR(0.00901f, get_current_rate(net), 1.0e-7f);
	ASSERT_NEAR(0.00901f, get_current_bias_rate(net), 1.0e-7f);
	ASSERT_FLOAT_EQ(0.937f, get_current_momentum(net));

	free_network(net);
}

TEST(DarknetScheduler, ParsesYolov9RuntimeRecipeFields)
{
	const std::string cfg =
		"[net]\n"
		"batch=1\n"
		"subdivisions=1\n"
		"width=32\n"
		"height=32\n"
		"channels=3\n"
		"max_batches=1000\n"
		"learning_rate=0.01\n"
		"lrf=0.01\n"
		"final_learning_rate=0.0001\n"
		"policy=linear_final\n"
		"warmup_iterations=100\n"
		"warmup_bias_lr=0.1\n"
		"warmup_momentum=0.8\n"
		"warmup_nonbias_lr_start=0.0\n"
		"close_mosaic_epochs=15\n"
		"close_mosaic_iteration=850\n"
		"augment_policy=yolov9\n"
		"hsv_h=0.015\n"
		"hsv_s=0.7\n"
		"hsv_v=0.4\n"
		"degrees=0.0\n"
		"translate=0.1\n"
		"yolov9_scale=0.9\n"
		"shear=0.0\n"
		"perspective=0.0\n"
		"flipud_prob=0.0\n"
		"fliplr_prob=0.5\n"
		"mosaic_prob=1.0\n"
		"mixup_prob=0.15\n"
		"copy_paste_prob=0.3\n"
		"\n"
		"[convolutional]\n"
		"filters=1\n"
		"size=1\n"
		"stride=1\n"
		"pad=1\n"
		"activation=linear\n";

	const auto path = write_temp_cfg("darknet_scheduler_linear_final.cfg", cfg);
	const GpuIndexGuard cpu_parse(-1);
	Darknet::Network net = parse_network_cfg_custom(path.string().c_str(), 1, 1);

	ASSERT_EQ(LINEAR_FINAL, net.policy);
	ASSERT_NEAR(0.0001f, net.final_learning_rate, 1.0e-8f);
	ASSERT_NEAR(0.01f, net.lrf, 1.0e-8f);
	ASSERT_EQ(100, net.warmup_iterations);
	ASSERT_NEAR(0.1f, net.warmup_bias_lr, 1.0e-8f);
	ASSERT_NEAR(0.8f, net.warmup_momentum, 1.0e-8f);
	ASSERT_EQ(15, net.close_mosaic_epochs);
	ASSERT_EQ(850, net.close_mosaic_iteration);
	ASSERT_EQ(1, net.augment_policy);
	ASSERT_NEAR(0.015f, net.hsv_h, 1.0e-8f);
	ASSERT_NEAR(0.7f, net.hsv_s, 1.0e-8f);
	ASSERT_NEAR(0.4f, net.hsv_v, 1.0e-8f);
	ASSERT_NEAR(0.1f, net.translate, 1.0e-8f);
	ASSERT_NEAR(0.9f, net.yolov9_scale, 1.0e-8f);
	ASSERT_NEAR(0.5f, net.fliplr_prob, 1.0e-8f);
	ASSERT_NEAR(1.0f, net.mosaic_prob, 1.0e-8f);
	ASSERT_NEAR(0.15f, net.mixup_prob, 1.0e-8f);
	ASSERT_NEAR(0.3f, net.copy_paste_prob, 1.0e-8f);

	free_network(net);
}
