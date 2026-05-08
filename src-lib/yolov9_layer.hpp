#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_yolov9_layer(int batch, int classes, int reg_max, int branch_count, int inference_branch, int input_count, int *input_layers, int *input_sizes, int *strides, int max_boxes, int total_points);
void resize_yolov9_layer(Darknet::Layer *l, Darknet::Network *net);
void forward_yolov9_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_yolov9_layer(Darknet::Layer & l, Darknet::NetworkState state);
int yolov9_num_detections(const Darknet::Network * net, const Darknet::Layer & l, float thresh);
int yolov9_num_detections_v3(Darknet::Network * net, const int index, const float thresh, Darknet::Output_Object_Cache & cache);
int yolov9_num_detections_batch(const Darknet::Network * net, const Darknet::Layer & l, float thresh, int batch);
int get_yolov9_detections(const Darknet::Network * net, const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter);
int get_yolov9_detections_batch(const Darknet::Network * net, const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter, int batch);

float yolov9_dfl_project(const float *logits, int reg_max);
Darknet::Box yolov9_dist2bbox(float anchor_x, float anchor_y, const float distances[4], float stride, int netw, int neth);

#ifdef DARKNET_GPU
void forward_yolov9_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_yolov9_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
