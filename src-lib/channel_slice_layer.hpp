#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_channel_slice_layer(int batch, int from, int w, int h, int c, int channel_start, int channel_count);
void forward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state);
void resize_channel_slice_layer(Darknet::Layer *l, Darknet::Network *net);

#ifdef DARKNET_GPU
void forward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
