#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "darknet_internal.hpp"

namespace
{
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
}

TEST(ChannelSlice, ParsesCfgSection)
{
	const auto cfg = write_temp_cfg(
		"darknet_channel_slice_test.cfg",
		"[net]\n"
		"batch=1\n"
		"subdivisions=1\n"
		"width=8\n"
		"height=8\n"
		"channels=3\n"
		"max_batches=1\n"
		"learning_rate=0.001\n"
		"\n"
		"[convolutional]\n"
		"filters=6\n"
		"size=1\n"
		"stride=1\n"
		"pad=1\n"
		"activation=linear\n"
		"\n"
		"[channel_slice]\n"
		"from=-1\n"
		"channel_start=2\n"
		"channel_count=3\n");

	Darknet::Network net = parse_network_cfg_custom(cfg.string().c_str(), 1, 1);
	ASSERT_EQ(net.n, 2);
	const Darknet::Layer & layer = net.layers[1];
	EXPECT_EQ(layer.type, Darknet::ELayerType::CHANNEL_SLICE);
	EXPECT_EQ(layer.input_layers[0], 0);
	EXPECT_EQ(layer.index, 0);
	EXPECT_EQ(layer.channel_start, 2);
	EXPECT_EQ(layer.channel_count, 3);
	EXPECT_EQ(layer.out_w, 8);
	EXPECT_EQ(layer.out_h, 8);
	EXPECT_EQ(layer.out_c, 3);
	EXPECT_EQ(layer.outputs, 8 * 8 * 3);
	free_network(net);
}

TEST(ChannelSlice, CopiesUnequalWindowAndAccumulatesDelta)
{
	Darknet::Layer source = { (Darknet::ELayerType)0 };
	source.out_w = source.w = 2;
	source.out_h = source.h = 2;
	source.out_c = source.c = 5;
	source.outputs = source.inputs = source.out_w * source.out_h * source.out_c;
	source.output = static_cast<float *>(xcalloc(source.outputs, sizeof(float)));
	source.delta = static_cast<float *>(xcalloc(source.outputs, sizeof(float)));
	for (int idx = 0; idx < source.outputs; ++idx)
	{
		source.output[idx] = static_cast<float>(idx + 1);
	}

	Darknet::Layer slice = make_channel_slice_layer(1, 0, source.out_w, source.out_h, source.out_c, 1, 3);

	Darknet::Layer layers[1] = { source };
	Darknet::Network net = { 0 };
	net.n = 1;
	net.layers = layers;
	Darknet::NetworkState state = { 0 };
	state.net = net;
	state.train = 1;

	forward_channel_slice_layer(slice, state);
	const std::vector<float> expected =
	{
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f, 12.0f,
		13.0f, 14.0f, 15.0f, 16.0f
	};
	for (std::size_t idx = 0; idx < expected.size(); ++idx)
	{
		EXPECT_FLOAT_EQ(slice.output[idx], expected[idx]);
		slice.delta[idx] = 1.0f + static_cast<float>(idx);
	}

	backward_channel_slice_layer(slice, state);
	for (int idx = 0; idx < source.outputs; ++idx)
	{
		const int channel = idx / 4;
		if (channel < 1 or channel > 3)
		{
			EXPECT_FLOAT_EQ(source.delta[idx], 0.0f);
		}
		else
		{
			const int offset = (channel - 1) * 4 + idx % 4;
			EXPECT_FLOAT_EQ(source.delta[idx], 1.0f + static_cast<float>(offset));
		}
	}

	free_layer(slice);
	free(source.output);
	free(source.delta);
}
