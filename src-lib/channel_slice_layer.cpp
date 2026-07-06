#include "darknet_internal.hpp"


Darknet::Layer make_channel_slice_layer(int batch, int from, int w, int h, int c, int channel_start, int channel_count)
{
	TAT(TATPARMS);

	if (channel_start < 0 or channel_count <= 0 or channel_start + channel_count > c)
	{
		darknet_fatal_error(
			DARKNET_LOC,
			"invalid channel_slice range start=%d count=%d for input channels=%d",
			channel_start,
			channel_count,
			c);
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::CHANNEL_SLICE;
	l.batch = batch;
	l.index = from;
	l.channel_start = channel_start;
	l.channel_count = channel_count;
	l.n = 1;
	l.w = l.out_w = w;
	l.h = l.out_h = h;
	l.c = c;
	l.out_c = channel_count;
	l.inputs = w * h * c;
	l.outputs = w * h * channel_count;
	l.input_layers = (int*)xcalloc(1, sizeof(int));
	l.input_sizes = (int*)xcalloc(1, sizeof(int));
	l.input_layers[0] = from;
	l.input_sizes[0] = l.inputs;
	l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
	l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

	l.forward = forward_channel_slice_layer;
	l.backward = backward_channel_slice_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_channel_slice_layer_gpu;
	l.backward_gpu = backward_channel_slice_layer_gpu;
	l.delta_gpu = cuda_make_array(l.delta, l.outputs * batch);
	l.output_gpu = cuda_make_array(l.output, l.outputs * batch);
#endif

	return l;
}


void resize_channel_slice_layer(Darknet::Layer *l, Darknet::Network *net)
{
	TAT(TATPARMS);

	Darknet::Layer & from = net->layers[l->input_layers[0]];
	if (l->channel_start < 0 or l->channel_count <= 0 or l->channel_start + l->channel_count > from.out_c)
	{
		darknet_fatal_error(
			DARKNET_LOC,
			"invalid channel_slice range start=%d count=%d for resized source channels=%d",
			l->channel_start,
			l->channel_count,
			from.out_c);
	}

	l->w = l->out_w = from.out_w;
	l->h = l->out_h = from.out_h;
	l->c = from.out_c;
	l->out_c = l->channel_count;
	l->inputs = from.outputs;
	l->outputs = l->out_w * l->out_h * l->out_c;
	l->input_sizes[0] = l->inputs;
	l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
	l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->outputs * l->batch);
	l->delta_gpu = cuda_make_array(l->delta, l->outputs * l->batch);
#endif
}


void forward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	int index = l.input_layers[0];
	int output_offset = 0;
	if (not state.train)
	{
		output_offset = resolve_inference_layer_output_offset(state.net, index);
		index = resolve_inference_layer_index(state.net, index);
	}

#ifdef DARKNET_USE_MPS
	mps_flush_deferred_output(&state.net.layers[index]);
#endif

	Darknet::Layer & from = state.net.layers[index];
	const int spatial = from.out_w * from.out_h;
	const int input_size = l.input_sizes[0];
	float *input = from.output + output_offset;

	for (int b = 0; b < l.batch; ++b)
	{
		for (int c = 0; c < l.channel_count; ++c)
		{
			float *src = input + b * input_size + (l.channel_start + c) * spatial;
			float *dst = l.output + b * l.outputs + c * spatial;
			copy_cpu(spatial, src, 1, dst, 1);
		}
	}
}


void backward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	Darknet::Layer & from = state.net.layers[l.input_layers[0]];
	const int spatial = from.out_w * from.out_h;
	const int input_size = l.input_sizes[0];

	for (int b = 0; b < l.batch; ++b)
	{
		for (int c = 0; c < l.channel_count; ++c)
		{
			float *src = l.delta + b * l.outputs + c * spatial;
			float *dst = from.delta + b * input_size + (l.channel_start + c) * spatial;
			axpy_cpu(spatial, 1.0f, src, 1, dst, 1);
		}
	}
}


#ifdef DARKNET_GPU
void forward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	int index = l.input_layers[0];
	int output_offset = 0;
	if (not state.train)
	{
		output_offset = resolve_inference_layer_output_offset(state.net, index);
		index = resolve_inference_layer_index(state.net, index);
	}

	Darknet::Layer & from = state.net.layers[index];
	const int spatial = from.out_w * from.out_h;
	const int input_size = l.input_sizes[0];
	float *input = from.output_gpu + output_offset;

	for (int b = 0; b < l.batch; ++b)
	{
		for (int c = 0; c < l.channel_count; ++c)
		{
			float *src = input + b * input_size + (l.channel_start + c) * spatial;
			float *dst = l.output_gpu + b * l.outputs + c * spatial;
			memcpy_ongpu(dst, src, spatial * sizeof(float));
		}
	}
}


void backward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	Darknet::Layer & from = state.net.layers[l.input_layers[0]];
	const int spatial = from.out_w * from.out_h;
	const int input_size = l.input_sizes[0];

	for (int b = 0; b < l.batch; ++b)
	{
		for (int c = 0; c < l.channel_count; ++c)
		{
			float *src = l.delta_gpu + b * l.outputs + c * spatial;
			float *dst = from.delta_gpu + b * input_size + (l.channel_start + c) * spatial;
			axpy_ongpu(spatial, 1.0f, src, 1, dst, 1);
		}
	}
}
#endif
