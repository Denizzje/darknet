#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	bool is_repconvn_conv(const Darknet::Layer & l, const int size)
	{
		return l.type == Darknet::ELayerType::CONVOLUTIONAL
			and l.size == size
			and l.stride_x == 1
			and l.stride_y == 1
			and l.dilation == 1
			and l.activation == LINEAR
			and l.batch_normalize == 0
			and l.groups > 0
			and l.weights != nullptr
			and l.biases != nullptr;
	}

	bool is_inference_optimizer_disabled()
	{
		const char * env = std::getenv("DARKNET_DISABLE_INFERENCE_OPTIMIZER");
		return env != nullptr and env[0] != '\0' and std::string(env) != "0";
	}

	bool env_flag_is_set(const char * name)
	{
		const char * env = std::getenv(name);
		return env != nullptr and env[0] != '\0' and std::string(env) != "0";
	}

	bool is_inference_cuda_graph_disabled()
	{
		return env_flag_is_set("DARKNET_DISABLE_INFERENCE_CUDA_GRAPH")
			or env_flag_is_set("DARKNET_DISABLE_CUDA_GRAPH")
			or env_flag_is_set("DARKNET_YOLOV9_PACK_INFERENCE_OUTPUT")
			or env_flag_is_set("DARKNET_YOLOV9_PULL_IN_FORWARD");
	}

	bool is_route_1x1_fusion_disabled()
	{
		return env_flag_is_set("DARKNET_DISABLE_ROUTE_1X1_FUSION")
			or not env_flag_is_set("DARKNET_ENABLE_ROUTE_1X1_FUSION");
	}

	bool is_yolov9_direct_head_disabled()
	{
		return env_flag_is_set("DARKNET_DISABLE_YOLOV9_DIRECT_HEAD")
			or not env_flag_is_set("DARKNET_ENABLE_YOLOV9_DIRECT_HEAD");
	}

	void mark_inference_alias(Darknet::Network & net, const int layer_index, const int alias_layer, const int output_offset = 0)
	{
		if (layer_index < 0 or layer_index >= net.n or alias_layer < 0 or alias_layer >= net.n or layer_index == alias_layer)
		{
			return;
		}

		Darknet::Layer & l = net.layers[layer_index];
		if (not l.inference_skip)
		{
			net.inference_optimizer_skipped_layers += 1;
		}
		l.inference_skip = 1;
		l.inference_alias_layer = alias_layer;
		l.inference_alias_output_offset = output_offset;
	}

	bool repconvn_pattern_matches(const Darknet::Network & net, const int conv3_idx)
	{
		if (conv3_idx < 1 or conv3_idx + 3 >= net.n)
		{
			return false;
		}

		const Darknet::Layer & conv3 = net.layers[conv3_idx];
		const Darknet::Layer & route = net.layers[conv3_idx + 1];
		const Darknet::Layer & conv1 = net.layers[conv3_idx + 2];
		const Darknet::Layer & shortcut = net.layers[conv3_idx + 3];

		if (not is_repconvn_conv(conv3, 3) or not is_repconvn_conv(conv1, 1))
		{
			return false;
		}
		if (conv3.pad == 0)
		{
			return false;
		}
		if (conv3.n != conv1.n or conv3.c != conv1.c or conv3.groups != conv1.groups)
		{
			return false;
		}
		if (conv3.out_w != conv1.out_w or conv3.out_h != conv1.out_h or conv3.out_c != conv1.out_c)
		{
			return false;
		}
		if (route.type != Darknet::ELayerType::ROUTE or route.n != 1 or route.groups != 1)
		{
			return false;
		}
		if (route.input_layers == nullptr or route.input_layers[0] != conv3_idx - 1)
		{
			return false;
		}
		if (shortcut.type != Darknet::ELayerType::SHORTCUT or shortcut.n != 1 or shortcut.activation != SWISH)
		{
			return false;
		}
		if (shortcut.input_layers == nullptr or shortcut.input_layers[0] != conv3_idx)
		{
			return false;
		}
		if (shortcut.outputs != conv3.outputs or shortcut.outputs != conv1.outputs)
		{
			return false;
		}

		return true;
	}

	void ensure_inference_activation_buffer(Darknet::Layer & l)
	{
		if (l.activation != SWISH and l.activation != MISH and l.activation != HARD_MISH)
		{
			return;
		}

		const size_t elements = static_cast<size_t>(l.batch) * static_cast<size_t>(l.outputs);
		if (l.activation_input == nullptr)
		{
			l.activation_input = static_cast<float *>(xcalloc(elements, sizeof(float)));
		}

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0 and l.activation_input_gpu == nullptr)
		{
			l.activation_input_gpu = cuda_make_array(l.activation_input, elements);
		}
#endif
	}

	void fuse_repconvn_block(Darknet::Network & net, const int conv3_idx)
	{
		Darknet::Layer & conv3 = net.layers[conv3_idx];
		Darknet::Layer & conv1 = net.layers[conv3_idx + 2];

		const int in_per_group = conv3.c / conv3.groups;
		const int conv3_filter_size = conv3.size * conv3.size * in_per_group;
		const int conv1_filter_size = conv1.size * conv1.size * in_per_group;
		const int center = (conv3.size / 2) * conv3.size + (conv3.size / 2);

		for (int filter = 0; filter < conv3.n; ++filter)
		{
			const int conv3_filter_offset = filter * conv3_filter_size;
			const int conv1_filter_offset = filter * conv1_filter_size;
			for (int channel = 0; channel < in_per_group; ++channel)
			{
				const int conv3_weight_index = conv3_filter_offset + channel * conv3.size * conv3.size + center;
				const int conv1_weight_index = conv1_filter_offset + channel;
				conv3.weights[conv3_weight_index] += conv1.weights[conv1_weight_index];
			}
			conv3.biases[filter] += conv1.biases[filter];
		}

		conv3.activation = SWISH;
		ensure_inference_activation_buffer(conv3);

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0 and conv3.weights_gpu != nullptr)
		{
			push_convolutional_layer(conv3);
		}
#endif

		mark_inference_alias(net, conv3_idx + 1, conv3_idx - 1);
		mark_inference_alias(net, conv3_idx + 2, conv3_idx);
		mark_inference_alias(net, conv3_idx + 3, conv3_idx);
		net.inference_optimizer_fused_repconvn += 1;
	}

	void remap_layer_input(Darknet::Network & net, Darknet::Layer & l, const int input_slot)
	{
		if (l.input_layers == nullptr)
		{
			return;
		}

		const int old_index = l.input_layers[input_slot];
		const int new_index = resolve_inference_layer_index(net, old_index);
		const int output_offset = resolve_inference_layer_output_offset(net, old_index);
		if (new_index < 0 or new_index >= net.n)
		{
			return;
		}
		if (old_index >= 0
			and old_index < net.n
			and net.layers[old_index].inference_skip
			and (output_offset != 0 or net.layers[old_index].outputs != net.layers[new_index].outputs))
		{
			return;
		}
		if (new_index == old_index)
		{
			return;
		}

		l.input_layers[input_slot] = new_index;
		if (l.input_sizes != nullptr)
		{
			l.input_sizes[input_slot] = net.layers[new_index].outputs;
		}
	}

	void refresh_shortcut_pointers(Darknet::Network & net, Darknet::Layer & l)
	{
		if (l.type != Darknet::ELayerType::SHORTCUT or l.input_layers == nullptr)
		{
			return;
		}

		for (int input = 0; input < l.n; ++input)
		{
			const int index = l.input_layers[input];
			const int resolved_index = resolve_inference_layer_index(net, index);
			const int output_offset = resolve_inference_layer_output_offset(net, index);
			if (l.layers_output != nullptr)
			{
				l.layers_output[input] = net.layers[resolved_index].output + output_offset;
			}
			if (l.layers_delta != nullptr)
			{
				l.layers_delta[input] = net.layers[resolved_index].delta + output_offset;
			}
		}

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			if (l.input_sizes_gpu != nullptr and l.input_sizes != nullptr)
			{
				memcpy_ongpu(l.input_sizes_gpu, l.input_sizes, l.n * sizeof(int));
			}
			if (l.layers_output_gpu != nullptr)
			{
				std::vector<float *> output_ptrs(l.n, nullptr);
				for (int input = 0; input < l.n; ++input)
				{
					const int index = l.input_layers[input];
					const int resolved_index = resolve_inference_layer_index(net, index);
					const int output_offset = resolve_inference_layer_output_offset(net, index);
					output_ptrs[input] = net.layers[resolved_index].output_gpu + output_offset;
				}
				memcpy_ongpu(l.layers_output_gpu, output_ptrs.data(), l.n * sizeof(float *));
			}
			if (l.layers_delta_gpu != nullptr)
			{
				std::vector<float *> delta_ptrs(l.n, nullptr);
				for (int input = 0; input < l.n; ++input)
				{
					const int index = l.input_layers[input];
					const int resolved_index = resolve_inference_layer_index(net, index);
					const int output_offset = resolve_inference_layer_output_offset(net, index);
					delta_ptrs[input] = net.layers[resolved_index].delta_gpu + output_offset;
				}
				memcpy_ongpu(l.layers_delta_gpu, delta_ptrs.data(), l.n * sizeof(float *));
			}
		}
#endif
	}

	void remap_inference_alias_consumers(Darknet::Network & net)
	{
		for (int i = 0; i < net.n; ++i)
		{
			Darknet::Layer & l = net.layers[i];
			if (l.input_layers != nullptr)
			{
				for (int input = 0; input < l.n; ++input)
				{
					remap_layer_input(net, l, input);
				}
				refresh_shortcut_pointers(net, l);
			}
		}
	}

	int count_inference_skipped_layers(const Darknet::Network & net)
	{
		int skipped = 0;
		for (int i = 0; i < net.n; ++i)
		{
			if (net.layers[i].inference_skip)
			{
				skipped += 1;
			}
		}
		return skipped;
	}

	std::vector<int> count_explicit_input_consumers(const Darknet::Network & net)
	{
		std::vector<int> consumers(net.n, 0);
		for (int i = 0; i < net.n; ++i)
		{
			const Darknet::Layer & l = net.layers[i];
			if (l.input_layers == nullptr)
			{
				continue;
			}
			for (int input = 0; input < l.n; ++input)
			{
				const int index = l.input_layers[input];
				if (index >= 0 and index < net.n)
				{
					consumers[index] += 1;
				}
			}
		}
		return consumers;
	}

	bool can_fuse_route_concat_1x1(const Darknet::Network & net, const int conv_idx, const std::vector<int> & explicit_consumers)
	{
#ifndef DARKNET_GPU
		return false;
#else
		if (is_route_1x1_fusion_disabled() or conv_idx <= 0 or conv_idx >= net.n)
		{
			return false;
		}
		if (cfg_and_state.gpu_index < 0)
		{
			return false;
		}

		const Darknet::Layer & route = net.layers[conv_idx - 1];
		const Darknet::Layer & conv = net.layers[conv_idx];
		if (route.inference_skip or route.type != Darknet::ELayerType::ROUTE or route.n <= 1 or route.groups != 1 or route.input_layers == nullptr or route.input_sizes == nullptr)
		{
			return false;
		}
		if (explicit_consumers[conv_idx - 1] != 0)
		{
			return false;
		}
		if (conv.type != Darknet::ELayerType::CONVOLUTIONAL or conv.batch != 1 or conv.size != 1 or conv.stride_x != 1 or conv.stride_y != 1 or conv.dilation != 1 or conv.groups != 1)
		{
			return false;
		}
		if (conv.batch_normalize != 0 or conv.weights_gpu == nullptr or conv.biases_gpu == nullptr)
		{
			return false;
		}
		if (conv.activation != LINEAR and conv.activation != SWISH)
		{
			return false;
		}
		if (route.out_w != conv.w or route.out_h != conv.h or route.out_c != conv.c or route.outputs != conv.inputs)
		{
			return false;
		}
		for (int input = 0; input < route.n; ++input)
		{
			const int index = route.input_layers[input];
			if (index < 0 or index >= net.n)
			{
				return false;
			}
			const Darknet::Layer & from = net.layers[index];
			if (from.out_w != route.out_w or from.out_h != route.out_h or from.outputs != route.input_sizes[input])
			{
				return false;
			}
		}
		return true;
#endif
	}

	void fuse_route_concat_1x1(Darknet::Network & net, const int conv_idx)
	{
		Darknet::Layer & route = net.layers[conv_idx - 1];
		Darknet::Layer & conv = net.layers[conv_idx];
		conv.inference_fused_route_layer = conv_idx - 1;
		conv.inference_fused_route_input_count = route.n;
		conv.inference_fused_route_input_sizes = static_cast<int *>(xcalloc(route.n, sizeof(int)));
		conv.inference_fused_route_layers_output = static_cast<float **>(xcalloc(route.n, sizeof(float *)));

		std::vector<float *> gpu_outputs(route.n, nullptr);
		for (int input = 0; input < route.n; ++input)
		{
			const int source_index = route.input_layers[input];
			const int resolved_index = resolve_inference_layer_index(net, source_index);
			const int output_offset = resolve_inference_layer_output_offset(net, source_index);
			conv.inference_fused_route_input_sizes[input] = route.input_sizes[input];
			conv.inference_fused_route_layers_output[input] = net.layers[resolved_index].output + output_offset;
#ifdef DARKNET_GPU
			gpu_outputs[input] = net.layers[resolved_index].output_gpu + output_offset;
#endif
		}

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			conv.inference_fused_route_input_sizes_gpu = cuda_make_int_array_new_api(conv.inference_fused_route_input_sizes, route.n);
			conv.inference_fused_route_layers_output_gpu = reinterpret_cast<float **>(cuda_make_array_pointers(reinterpret_cast<void **>(gpu_outputs.data()), route.n));
		}
#endif

		const int alias_index = resolve_inference_layer_index(net, route.input_layers[0]);
		const int output_offset = resolve_inference_layer_output_offset(net, route.input_layers[0]);
		mark_inference_alias(net, conv_idx - 1, alias_index, output_offset);
		net.inference_optimizer_fused_route_1x1 += 1;
	}

	void fuse_route_concat_1x1_consumers(Darknet::Network & net)
	{
		const std::vector<int> explicit_consumers = count_explicit_input_consumers(net);
		for (int i = 1; i < net.n; ++i)
		{
			if (can_fuse_route_concat_1x1(net, i, explicit_consumers))
			{
				fuse_route_concat_1x1(net, i);
			}
		}
	}

	bool yolov9_head_route_matches(const Darknet::Network & net, const Darknet::Layer & yolo, const int route_idx)
	{
		if (is_yolov9_direct_head_disabled() or route_idx < 0 or route_idx >= net.n)
		{
			return false;
		}
		const Darknet::Layer & route = net.layers[route_idx];
		if (route.inference_skip or route.type != Darknet::ELayerType::ROUTE or route.n != 2 or route.groups != 1 or route.input_layers == nullptr)
		{
			return false;
		}
		const Darknet::Layer & box = net.layers[route.input_layers[0]];
		const Darknet::Layer & cls = net.layers[route.input_layers[1]];
		if (box.out_w != cls.out_w or box.out_h != cls.out_h)
		{
			return false;
		}
		if (box.out_c != 4 * yolo.reg_max or cls.out_c != yolo.classes)
		{
			return false;
		}
		return route.out_w == box.out_w
			and route.out_h == box.out_h
			and route.outputs == box.outputs + cls.outputs;
	}

	void mark_direct_yolov9_head_routes(Darknet::Network & net)
	{
		for (int i = 0; i < net.n; ++i)
		{
			Darknet::Layer & yolo = net.layers[i];
			if (yolo.type != Darknet::ELayerType::YOLOV9 or yolo.input_layers == nullptr)
			{
				continue;
			}
			const int branch = std::max(0, std::min(yolo.inference_branch, yolo.branch_count - 1));
			const int branch_offset = branch * yolo.n;
			for (int scale = 0; scale < yolo.n; ++scale)
			{
				const int slot = branch_offset + scale;
				if (slot < 0 or slot >= yolo.total)
				{
					continue;
				}
				const int route_idx = yolo.input_layers[slot];
				if (not yolov9_head_route_matches(net, yolo, route_idx))
				{
					continue;
				}
				Darknet::Layer & route = net.layers[route_idx];
				route.inference_direct_yolov9_head = 1;
				const int alias_index = resolve_inference_layer_index(net, route.input_layers[0]);
				const int output_offset = resolve_inference_layer_output_offset(net, route.input_layers[0]);
				mark_inference_alias(net, route_idx, alias_index, output_offset);
				net.inference_optimizer_direct_yolov9_heads += 1;
			}
		}
	}

	void maybe_log_inference_optimizer_summary(const Darknet::Network & net)
	{
		const char * verbose_env = std::getenv("DARKNET_INFERENCE_OPTIMIZER_VERBOSE");
		const bool verbose_env_enabled = verbose_env != nullptr and verbose_env[0] != '\0' and std::string(verbose_env) != "0";
		if (not cfg_and_state.is_verbose and not verbose_env_enabled)
		{
			return;
		}

		*cfg_and_state.output
			<< "Darknet inference optimizer: fused " << net.inference_optimizer_fused_repconvn
			<< " RepConvN blocks, aliased " << net.inference_optimizer_aliased_routes
			<< " route layers, fused " << net.inference_optimizer_fused_route_1x1
			<< " route->1x1 convs, direct " << net.inference_optimizer_direct_yolov9_heads
			<< " YOLOv9 heads, disabled half inference " << net.inference_optimizer_disabled_cudnn_half
			<< ", pruned " << net.inference_optimizer_pruned_layers
			<< " branch layers, skipped " << net.inference_optimizer_skipped_layers
			<< " of " << net.n << " layers." << std::endl;
	}

	bool has_multibranch_yolov9_output(const Darknet::Network & net)
	{
		for (int i = 0; i < net.n; ++i)
		{
			const Darknet::Layer & l = net.layers[i];
			if (l.type == Darknet::ELayerType::YOLOV9 and l.branch_count > 1)
			{
				return true;
			}
		}
		return false;
	}

	bool has_yolov9_output(const Darknet::Network & net)
	{
		for (int i = 0; i < net.n; ++i)
		{
			if (net.layers[i].type == Darknet::ELayerType::YOLOV9)
			{
				return true;
			}
		}
		return false;
	}

	void mark_inference_reachable_layer(const Darknet::Network & net, int layer_index, std::vector<unsigned char> & reachable)
	{
		layer_index = resolve_inference_layer_index(net, layer_index);
		if (layer_index < 0 or layer_index >= net.n or reachable[layer_index])
		{
			return;
		}

		reachable[layer_index] = 1;
		const Darknet::Layer & l = net.layers[layer_index];

		if (l.type == Darknet::ELayerType::YOLOV9)
		{
			const int branch = std::max(0, std::min(l.inference_branch, l.branch_count - 1));
			const int first_slot = branch * l.n;
			const int last_slot = std::min(first_slot + l.n, l.total);
			for (int slot = first_slot; slot < last_slot; ++slot)
			{
				if (l.input_layers != nullptr)
				{
					mark_inference_reachable_layer(net, l.input_layers[slot], reachable);
				}
			}
			return;
		}

		if (l.type == Darknet::ELayerType::ROUTE)
		{
			for (int input = 0; input < l.n; ++input)
			{
				if (l.input_layers != nullptr)
				{
					mark_inference_reachable_layer(net, l.input_layers[input], reachable);
				}
			}
			return;
		}

		if (l.type == Darknet::ELayerType::SHORTCUT)
		{
			mark_inference_reachable_layer(net, layer_index - 1, reachable);
			for (int input = 0; input < l.n; ++input)
			{
				if (l.input_layers != nullptr)
				{
					mark_inference_reachable_layer(net, l.input_layers[input], reachable);
				}
			}
			return;
		}

		if (layer_index > 0)
		{
			mark_inference_reachable_layer(net, layer_index - 1, reachable);
		}
	}

	void prune_unreachable_yolov9_branches(Darknet::Network & net)
	{
		if (not has_multibranch_yolov9_output(net))
		{
			return;
		}

		std::vector<unsigned char> reachable(net.n, 0);
		for (int i = net.n - 1; i >= 0; --i)
		{
			if (net.layers[i].type == Darknet::ELayerType::YOLOV9)
			{
				mark_inference_reachable_layer(net, i, reachable);
			}
		}

		for (int i = 0; i < net.n; ++i)
		{
			if (reachable[i] or net.layers[i].inference_skip)
			{
				continue;
			}
			const int alias_index = resolve_inference_layer_index(net, i - 1);
			if (alias_index >= 0 and alias_index < net.n)
			{
				mark_inference_alias(net, i, alias_index);
				net.inference_optimizer_pruned_layers += 1;
			}
		}
	}

	void alias_single_input_routes(Darknet::Network & net)
	{
		for (int i = 0; i < net.n; ++i)
		{
			Darknet::Layer & l = net.layers[i];
			if (l.inference_skip or l.type != Darknet::ELayerType::ROUTE or l.n != 1 or l.input_layers == nullptr)
			{
				continue;
			}

			const int input_index = l.input_layers[0];
			const int alias_index = resolve_inference_layer_index(net, input_index);
			if (alias_index < 0 or alias_index >= net.n)
			{
				continue;
			}

			const int base_offset = resolve_inference_layer_output_offset(net, input_index);
			if (l.groups == 1)
			{
				mark_inference_alias(net, i, alias_index, base_offset);
				net.inference_optimizer_aliased_routes += 1;
				continue;
			}

			if (l.batch != 1 or l.groups <= 1 or l.group_id < 0 or l.group_id >= l.groups or l.input_sizes[0] % l.groups != 0)
			{
				continue;
			}

			const int part_input_size = l.input_sizes[0] / l.groups;
			if (part_input_size != l.outputs)
			{
				continue;
			}
			mark_inference_alias(net, i, alias_index, base_offset + part_input_size * l.group_id);
			net.inference_optimizer_aliased_routes += 1;
		}
	}
}


Darknet::NetworkDetails::NetworkDetails()
{
	TAT(TATPARMS);

	detection_threshold						= 0.25f;
	non_maximal_suppression_threshold		= 0.45f;

	fix_out_of_bound_normalized_coordinates	= true;

	cv_line_type							= cv::LineTypes::LINE_4;
	cv_font_face							= cv::HersheyFonts::FONT_HERSHEY_PLAIN;
	cv_font_thickness						= 1;
	cv_font_scale							= 1.0;

	bounding_boxes_with_rounded_corners		= false;
	bounding_boxes_corner_roundness			= 0.5f;

	annotate_draw_bb						= true;
	annotate_draw_label						= true;

	return;
}


int64_t get_current_iteration(const Darknet::Network & net)
{
	TAT(TATPARMS);

	return *net.cur_iteration;
}


int get_current_batch(const Darknet::Network & net)
{
	TAT(TATPARMS);

	const int batch_num = (*net.seen) / (net.batch * net.subdivisions);

	return batch_num;
}


void reset_network_state(Darknet::Network *net, int b)
{
	TAT(TATPARMS);

	for (int i = 0; i < net->n; ++i)
	{
#ifdef DARKNET_GPU
		Darknet::Layer & l = net->layers[i];
		if (l.state_gpu)
		{
			fill_ongpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
		}

		if (l.h_gpu)
		{
			fill_ongpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
		}
#endif
	}
}


void reset_rnn(DarknetNetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);

	reset_network_state(net, 0);
}


float get_current_seq_subdivisions(const Darknet::Network & net)
{
	TAT(TATPARMS);

	int sequence_subdivisions = net.init_sequential_subdivisions;

	if (net.num_steps > 0)
	{
		int batch_num = get_current_batch(net);

		for (int i = 0; i < net.num_steps; ++i)
		{
			if (net.steps[i] > batch_num)
			{
				break;
			}
			sequence_subdivisions *= net.seq_scales[i];
		}
	}

	if (sequence_subdivisions < 1)
	{
		sequence_subdivisions = 1;
	}

	if (sequence_subdivisions > net.subdivisions)
	{
		sequence_subdivisions = net.subdivisions;
	}

	return sequence_subdivisions;
}


int get_sequence_value(const Darknet::Network & net)
{
	TAT(TATPARMS);

	int sequence = 1;
	if (net.sequential_subdivisions != 0)
	{
		sequence = net.subdivisions / net.sequential_subdivisions;
	}
	if (sequence < 1)
	{
		sequence = 1;
	}

	return sequence;
}


namespace
{
	float clamp01(const float value)
	{
		if (value < 0.0f)
		{
			return 0.0f;
		}
		if (value > 1.0f)
		{
			return 1.0f;
		}
		return value;
	}

	float lerp(const float start, const float end, const float fraction)
	{
		return start + (end - start) * clamp01(fraction);
	}

	float get_current_rate_without_warmup(const Darknet::Network & net, const int batch_num)
	{
		int i;
		float rate;
		switch (net.policy)
		{
			case CONSTANT:
				return net.learning_rate;
			case STEP:
				return net.learning_rate * pow(net.scale, batch_num/net.step);
			case STEPS:
				rate = net.learning_rate;
				for (i = 0; i < net.num_steps; ++i)
				{
					if (net.steps[i] > batch_num)
					{
						return rate;
					}
					rate *= net.scales[i];
					//if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
				}
				return rate;
			case EXP:
				return net.learning_rate * pow(net.gamma, batch_num);
			case POLY:
				return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
			case LINEAR_FINAL:
			{
				const float fraction = net.max_batches > 0 ? (float)batch_num / (float)net.max_batches : 1.0f;
				return lerp(net.learning_rate, net.final_learning_rate, fraction);
			}
			case RANDOM:
				return net.learning_rate * pow(rand_uniform(0.0f, 1.0f), net.power);
			case SIG:
				return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
			case SGDR:
			{
				int last_iteration_start = 0;
				int cycle_size = net.batches_per_cycle;
				while ((last_iteration_start + cycle_size) < batch_num)
				{
					last_iteration_start += cycle_size;
					cycle_size *= net.batches_cycle_mult;
				}
				rate = net.learning_rate_min +
					0.5*(net.learning_rate - net.learning_rate_min)
					* (1. + cos((float)(batch_num - last_iteration_start) * M_PI / cycle_size));

				return rate;
			}
			default:
				*cfg_and_state.output << "Unknown policy (" << net.policy << ")" << std::endl;
				return net.learning_rate;
		}
	}

	float get_warmup_fraction(const Darknet::Network & net, const int batch_num)
	{
		if (net.warmup_iterations <= 0 or batch_num >= net.warmup_iterations)
		{
			return 1.0f;
		}
		return (float)batch_num / (float)net.warmup_iterations;
	}
}


float get_current_rate(const Darknet::Network & net)
{
	TAT(TATPARMS);

	int batch_num = get_current_batch(net);
	if (net.warmup_iterations > 0 and batch_num < net.warmup_iterations)
	{
		const float target_rate = get_current_rate_without_warmup(net, batch_num);
		return lerp(net.warmup_nonbias_lr_start, target_rate, get_warmup_fraction(net, batch_num));
	}
	if (batch_num < net.burn_in)
	{
		return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
	}
	return get_current_rate_without_warmup(net, batch_num);
}


float get_current_bias_rate(const Darknet::Network & net)
{
	TAT(TATPARMS);

	const int batch_num = get_current_batch(net);
	if (net.warmup_iterations > 0 and batch_num < net.warmup_iterations)
	{
		const float target_rate = get_current_rate_without_warmup(net, batch_num);
		return lerp(net.warmup_bias_lr, target_rate, get_warmup_fraction(net, batch_num));
	}
	return get_current_rate(net);
}


float get_current_momentum(const Darknet::Network & net)
{
	TAT(TATPARMS);

	const int batch_num = get_current_batch(net);
	if (net.warmup_iterations > 0 and batch_num < net.warmup_iterations)
	{
		return lerp(net.warmup_momentum, net.momentum, get_warmup_fraction(net, batch_num));
	}
	return net.momentum;
}


Darknet::Network make_network(int n)
{
	TAT(TATPARMS);

	Darknet::Network net = {0};
	net.n = n;
	net.layers = (Darknet::Layer*)xcalloc(net.n, sizeof(Darknet::Layer));
	net.seen = (uint64_t*)xcalloc(1, sizeof(uint64_t));
	net.cuda_graph_ready = (int*)xcalloc(1, sizeof(int));
	net.badlabels_reject_threshold = (float*)xcalloc(1, sizeof(float));
	net.delta_rolling_max = (float*)xcalloc(1, sizeof(float));
	net.delta_rolling_avg = (float*)xcalloc(1, sizeof(float));
	net.delta_rolling_std = (float*)xcalloc(1, sizeof(float));
	net.cur_iteration = (int*)xcalloc(1, sizeof(int));
	net.total_bbox = (int*)xcalloc(1, sizeof(int));
	net.rewritten_bbox = (int*)xcalloc(1, sizeof(int));
	*net.rewritten_bbox = *net.total_bbox = 0;
#ifdef DARKNET_GPU
	net.input_gpu = (float**)xcalloc(1, sizeof(float*));
	net.truth_gpu = (float**)xcalloc(1, sizeof(float*));

	net.input16_gpu = (float**)xcalloc(1, sizeof(float*));
	net.output16_gpu = (float**)xcalloc(1, sizeof(float*));
	net.max_input16_size = (size_t*)xcalloc(1, sizeof(size_t));
	net.max_output16_size = (size_t*)xcalloc(1, sizeof(size_t));
#endif

	net.details = new Darknet::NetworkDetails;

	return net;
}


void forward_network(Darknet::Network & net, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	state.workspace = net.workspace;

	for (int i = 0; i < net.n; ++i)
	{
		state.index = i;
		Darknet::Layer & l = net.layers[i];
		if (not state.train and l.inference_skip)
		{
			const int alias_index = resolve_inference_layer_index(net, i);
			const int output_offset = resolve_inference_layer_output_offset(net, i);
			if (alias_index >= 0 and alias_index < net.n)
			{
				state.input = net.layers[alias_index].output + output_offset;
			}
			continue;
		}
		if (l.delta && state.train && l.train)
		{
			scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
		}
#ifdef DARKNET_USE_MPS
		if (i > 0)
		{
			Darknet::Layer & prev = net.layers[i - 1];
			if (mps_is_output_deferred(&prev) and not mps_layer_can_run(l, state.train))
			{
				mps_flush_output_if_needed(&prev, prev.output);
			}
		}
#endif
		l.forward(l, state);
		state.input = l.output;
	}
#ifdef DARKNET_USE_MPS
	if (net.n > 0)
	{
		Darknet::Layer & last = net.layers[net.n - 1];
		if (mps_is_output_deferred(&last))
		{
			mps_flush_output_if_needed(&last, last.output);
		}
	}
#endif
}


void update_network(Darknet::Network & net)
{
	TAT(TATPARMS);

	int update_batch = net.batch * net.subdivisions;
	float rate = get_current_rate(net);
	float bias_rate = get_current_bias_rate(net);
	float momentum = get_current_momentum(net);

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.train == 0)
		{
			continue;
		}

		if (l.update)
		{
			l.use_current_update_rates = 1;
			l.current_learning_rate = rate * l.learning_rate_scale;
			l.current_bias_learning_rate = bias_rate * l.learning_rate_scale;
			l.update(l, update_batch, rate, momentum, net.decay);
			l.use_current_update_rates = 0;
		}
	}
}


float *get_network_output(Darknet::Network & net)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		return get_network_output_gpu(net);
	}
#endif

	int i;
	for (i = net.n-1; i > 0; --i)
	{
		if (net.layers[i].type != Darknet::ELayerType::COST)
		{
			break;
		}
	}
	const int output_offset = resolve_inference_layer_output_offset(net, i);
	i = resolve_inference_layer_index(net, i);

	return net.layers[i].output + output_offset;
}

int inference_optimizer_is_disabled()
{
	TAT(TATPARMS);

	return is_inference_optimizer_disabled() ? 1 : 0;
}

int resolve_inference_layer_index(const Darknet::Network & net, int layer_index)
{
	int guard = 0;
	while (layer_index >= 0 and layer_index < net.n and guard < net.n)
	{
		const Darknet::Layer & l = net.layers[layer_index];
		if (not l.inference_skip)
		{
			break;
		}
		if (l.inference_alias_layer < 0 or l.inference_alias_layer == layer_index)
		{
			break;
		}
		layer_index = l.inference_alias_layer;
		guard += 1;
	}
	return layer_index;
}

int resolve_inference_layer_output_offset(const Darknet::Network & net, int layer_index)
{
	int output_offset = 0;
	int guard = 0;
	while (layer_index >= 0 and layer_index < net.n and guard < net.n)
	{
		const Darknet::Layer & l = net.layers[layer_index];
		if (not l.inference_skip)
		{
			break;
		}
		output_offset += l.inference_alias_output_offset;
		if (l.inference_alias_layer < 0 or l.inference_alias_layer == layer_index)
		{
			break;
		}
		layer_index = l.inference_alias_layer;
		guard += 1;
	}
	return output_offset;
}

void optimize_network_for_inference(Darknet::Network & net)
{
	TAT(TATPARMS);

	static std::mutex optimizer_mutex;
	std::lock_guard<std::mutex> lock(optimizer_mutex);

	if (net.inference_optimized or is_inference_optimizer_disabled())
	{
		return;
	}

	net.inference_optimizer_skipped_layers = 0;
	net.inference_optimizer_fused_repconvn = 0;
	net.inference_optimizer_aliased_routes = 0;
	net.inference_optimizer_pruned_layers = 0;
	net.inference_optimizer_fused_route_1x1 = 0;
	net.inference_optimizer_direct_yolov9_heads = 0;
	net.inference_optimizer_disabled_cudnn_half = 0;
	net.inference_cuda_graph_enabled = 0;
	net.inference_cuda_graph_captured = 0;
	net.inference_cuda_graph_launches = 0;

	for (int i = 1; i + 3 < net.n; ++i)
	{
		if (repconvn_pattern_matches(net, i))
		{
			fuse_repconvn_block(net, i);
			i += 3;
		}
	}

	alias_single_input_routes(net);

	remap_inference_alias_consumers(net);
	prune_unreachable_yolov9_branches(net);
	remap_inference_alias_consumers(net);
	fuse_route_concat_1x1_consumers(net);
	mark_direct_yolov9_head_routes(net);
	net.inference_optimizer_skipped_layers = count_inference_skipped_layers(net);
	if (has_yolov9_output(net) and net.cudnn_half and not env_flag_is_set("DARKNET_ENABLE_YOLOV9_HALF_INFERENCE"))
	{
		net.cudnn_half = 0;
		net.inference_optimizer_disabled_cudnn_half = 1;
	}
	if (has_yolov9_output(net) and not is_inference_cuda_graph_disabled())
	{
		net.use_cuda_graph = 1;
		net.inference_cuda_graph_enabled = 1;
	}
	net.inference_optimized = 1;
	maybe_log_inference_optimizer_summary(net);
}


float get_network_cost(const Darknet::Network & net)
{
	TAT(TATPARMS);

	float sum = 0.0f;
	int count = 0;

	for (int i = 0; i < net.n; ++i)
	{
		if (net.layers[i].cost)
		{
			sum += net.layers[i].cost[0];
			++count;
		}
	}

	return sum/count;
}


void backward_network(Darknet::Network & net, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	float *original_input = state.input;
	float *original_delta = state.delta;
	state.workspace = net.workspace;

	for (int i = net.n-1; i >= 0; --i)
	{
		state.index = i;
		if (i == 0)
		{
			state.input = original_input;
			state.delta = original_delta;
		}
		else
		{
			Darknet::Layer & prev = net.layers[i-1];
			state.input = prev.output;
			state.delta = prev.delta;
		}

		Darknet::Layer & l = net.layers[i];
		if (l.stopbackward)
		{
			break;
		}

		if (l.onlyforward)
		{
			continue;
		}

		l.backward(l, state);
	}
}


float train_network_datum(Darknet::Network & net, float *x, float *y)
{
	TAT(TATPARMS);

	float error = 0.0f;

#ifdef DARKNET_GPU
	if(cfg_and_state.gpu_index >= 0)
	{
		error = train_network_datum_gpu(net, x, y);
	}
	else
	{
#endif

		Darknet::NetworkState state={0};
		*net.seen += net.batch;
		state.index = 0;
		state.net = net;
		state.input = x;
		state.delta = 0;
		state.truth = y;
		state.train = 1;
		forward_network(net, state);
		backward_network(net, state);
		error = get_network_cost(net);

#ifdef DARKNET_GPU
	}
#endif

	if (cfg_and_state.is_verbose and *(net.total_bbox) > 0)
	{
		*cfg_and_state.output
			<< "total_bbox=" << *(net.total_bbox)
			<< ", rewritten_bbox=" << 100.0f * float(*(net.rewritten_bbox)) / float(*(net.total_bbox))
			<< "%" << std::endl;
	}

	return error;
}


float train_network(Darknet::Network & net, data d)
{
	TAT(TATPARMS);

	return train_network_waitkey(net, d, 0);
}


float train_network_waitkey(Darknet::Network & net, data d, int wait_key)
{
	TAT_COMMENT(TATPARMS, "complicated");

	assert(d.X.rows % net.batch == 0);

	int batch = net.batch;
	int n = d.X.rows / batch;
	float* X = (float*)xcalloc(batch * d.X.cols, sizeof(float));
	float* y = (float*)xcalloc(batch * d.y.cols, sizeof(float));

	float sum = 0;
	for (int i = 0; i < n; ++i)
	{
		get_next_batch(d, batch, i*batch, X, y);
		net.current_subdivision = i;
		float err = train_network_datum(net, X, y);
		sum += err;
		if (wait_key)
		{
			cv::waitKey(5);
		}
	}

	(*net.cur_iteration) += 1;
#ifdef DARKNET_GPU
	update_network_gpu(net);
#else   // DARKNET_GPU
	update_network(net);
#endif  // DARKNET_GPU

	int ema_start_point = net.max_batches / 2;

	if (net.ema_alpha && (*net.cur_iteration) >= ema_start_point)
	{
		//int ema_period = (net.max_batches - ema_start_point - 1000) * (1.0 - net.ema_alpha);
		int ema_apply_point = net.max_batches - 1000;

		if (!is_ema_initialized(net))
		{
			ema_update(net, 0); // init EMA
			*cfg_and_state.output << "EMA initialization" << std::endl;
		}

		if ((*net.cur_iteration) == ema_apply_point)
		{
			ema_apply(net); // apply EMA (BN rolling mean/var recalculation is required)
			*cfg_and_state.output << "EMA apply" << std::endl;
		}
		else
		{
			if ((*net.cur_iteration) < ema_apply_point)// && (*net.cur_iteration) % ema_period == 0)
			{
				ema_update(net, net.ema_alpha); // update EMA
				*cfg_and_state.output << "EMA update, alpha=" << net.ema_alpha << std::endl;
			}
		}
	}

	int reject_stop_point = net.max_batches * 3 / 4;

	if ((*net.cur_iteration) < reject_stop_point &&
		net.weights_reject_freq &&
		(*net.cur_iteration) % net.weights_reject_freq == 0)
	{
		float sim_threshold = 0.4;
		reject_similar_weights(net, sim_threshold);
	}

	free(X);
	free(y);

	return (float)sum/(n*batch);
}


float train_network_batch(Darknet::Network net, data d, int n)
{
	TAT(TATPARMS);

	Darknet::NetworkState state={0};
	state.index = 0;
	state.net = net;
	state.train = 1;
	state.delta = 0;
	float sum = 0;
	int batch = 2;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < batch; ++j)
		{
			int index = rand_uint(0, d.X.rows - 1);
			state.input = d.X.vals[index];
			state.truth = d.y.vals[index];
			forward_network(net, state);
			backward_network(net, state);
			sum += get_network_cost(net);
		}
		update_network(net);
	}

	return (float)sum/(n*batch);
}


int recalculate_workspace_size(Darknet::Network * net)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	cuda_set_device(net->gpu_index);
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(net->workspace);
	}
#endif

	size_t workspace_size = 0;
	for (int i = 0; i < net->n; ++i)
	{
		Darknet::Layer & l = net->layers[i];

		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			l.workspace_size = get_convolutional_workspace_size(l);
		}
		else if (l.type == Darknet::ELayerType::CONNECTED)
		{
			l.workspace_size = get_connected_workspace_size(l);
		}

		if (l.workspace_size > workspace_size)
		{
			workspace_size = l.workspace_size;
		}
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		const auto workspace_to_allocate = workspace_size / sizeof(float) + 1;
		*cfg_and_state.output << std::endl << "allocating workspace: " << Darknet::size_to_IEC_string(workspace_to_allocate) << std::endl;
		net->workspace = cuda_make_array(0, workspace_to_allocate);
		*cfg_and_state.output << "CUDA allocate done!" << std::endl;
	}
	else
	{
		free(net->workspace);
		net->workspace = (float*)xcalloc(1, workspace_size);
	}
#else
	free(net->workspace);
	net->workspace = (float*)xcalloc(1, workspace_size);
#endif

	return 0;
}


void set_batch_network(Darknet::Network * net, int b)
{
	TAT(TATPARMS);

	net->batch = b;
	int i;
	for (i = 0; i < net->n; ++i)
	{
		net->layers[i].batch = b;

#ifdef CUDNN
		if(net->layers[i].type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			cudnn_convolutional_setup(net->layers + i, cudnn_fastest, 0);
		}
		else if (net->layers[i].type == Darknet::ELayerType::MAXPOOL)
		{
			cudnn_maxpool_setup(net->layers + i);
		}
#endif

	}
	recalculate_workspace_size(net); // recalculate workspace size
}


int resize_network(Darknet::Network * net, int w, int h)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	cuda_set_device(net->gpu_index);
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(net->workspace);
		if (net->input_gpu)
		{
			cuda_free(*net->input_gpu);
			*net->input_gpu = 0;
			cuda_free(*net->truth_gpu);
			*net->truth_gpu = 0;
		}

		if (net->input_state_gpu)
		{
			cuda_free(net->input_state_gpu);
		}

		if (net->input_pinned_cpu)
		{
			if (net->input_pinned_cpu_flag)
			{
				CHECK_CUDA(cudaFreeHost(net->input_pinned_cpu));
			}
			else
			{
				free(net->input_pinned_cpu);
			}
		}
	}
#endif

	//if(w == net->w && h == net->h) return 0;
	net->w = w;
	net->h = h;
	int inputs = 0;
	size_t workspace_size = 0;

	for (int i = 0; i < net->n; ++i)
	{
		Darknet::Layer & l = net->layers[i];

		switch (l.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:	resize_convolutional_layer(&l, w, h);		break;
			case Darknet::ELayerType::CRNN:				resize_crnn_layer(&l, w, h);				break;
			case Darknet::ELayerType::MAXPOOL:			resize_maxpool_layer(&l, w, h);				break;
			case Darknet::ELayerType::LOCAL_AVGPOOL:	resize_maxpool_layer(&l, w, h);				break;
			case Darknet::ELayerType::REGION:			resize_region_layer(&l, w, h);				break;
			case Darknet::ELayerType::YOLO:				resize_yolo_layer(&l, w, h);				break;
			case Darknet::ELayerType::YOLOV9:			resize_yolov9_layer(&l, net);				break;
			case Darknet::ELayerType::GAUSSIAN_YOLO:	resize_gaussian_yolo_layer(&l, w, h);		break;
			case Darknet::ELayerType::ROUTE:			resize_route_layer(&l, net);				break;
			case Darknet::ELayerType::CHANNEL_SLICE:	resize_channel_slice_layer(&l, net);		break;
			case Darknet::ELayerType::SHORTCUT:			resize_shortcut_layer(&l, w, h, net);		break;
			case Darknet::ELayerType::SCALE_CHANNELS:	resize_scale_channels_layer(&l, net);		break;
			case Darknet::ELayerType::SAM:				resize_sam_layer(&l, w, h);					break;
			case Darknet::ELayerType::UPSAMPLE:			resize_upsample_layer(&l, w, h);			break;
			case Darknet::ELayerType::REORG:			resize_reorg_layer(&l, w, h);				break;
			case Darknet::ELayerType::AVGPOOL:			resize_avgpool_layer(&l, w, h);				break;
			case Darknet::ELayerType::COST:				resize_cost_layer(&l, inputs);				break;
			case Darknet::ELayerType::DROPOUT:
			{
				resize_dropout_layer(&l, inputs);
				l.out_w			= l.w = w;
				l.out_h			= l.h = h;
				l.output		= net->layers[i - 1].output;
				l.delta			= net->layers[i - 1].delta;
#ifdef DARKNET_GPU
				l.output_gpu	= net->layers[i-1].output_gpu;
				l.delta_gpu		= net->layers[i-1].delta_gpu;
#endif
				break;
			}
			default:
			{
				darknet_fatal_error(DARKNET_LOC, "cannot resize layer type #%d", (int)l.type);
			}
		}

		if (l.workspace_size > workspace_size)
		{
			workspace_size = l.workspace_size;
		}

		inputs = l.outputs;
//		net->layers[i] = l;
		//if(l.type != DROPOUT)
		{
			w = l.out_w;
			h = l.out_h;
		}
		//if(l.type == AVGPOOL) break;
	}

	*cfg_and_state.output << "GPU #" << net->gpu_index << ": allocating workspace: " << Darknet::size_to_IEC_string(workspace_size);
#ifdef DARKNET_GPU
	const int size = get_network_input_size(*net) * net->batch;
	if (cfg_and_state.gpu_index >= 0)
	{
		net->workspace = cuda_make_array(0, workspace_size/sizeof(float) + 1);
		net->input_state_gpu = cuda_make_array(0, size);
		if (cudaSuccess == cudaHostAlloc((void**)&net->input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
		{
			net->input_pinned_cpu_flag = 1;
		}
		else
		{
			cudaError_t status = cudaGetLastError(); // reset CUDA-error
			*cfg_and_state.output << std::endl << "CUDA error #" << status << " (" << cudaGetErrorName(status) << ", " << cudaGetErrorString(status) << ")" << std::endl;
			net->input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
			net->input_pinned_cpu_flag = 0;
		}
	}
	else
	{
		free(net->workspace);
		net->workspace = (float*)xcalloc(1, workspace_size);
		if (!net->input_pinned_cpu_flag)
		{
			net->input_pinned_cpu = (float*)xrealloc(net->input_pinned_cpu, size * sizeof(float));
		}
	}
#else
	free(net->workspace);
	net->workspace = (float*)xcalloc(1, workspace_size);
#endif
	if (net->workspace == NULL)
	{
		*cfg_and_state.output << std::endl;
		darknet_fatal_error(DARKNET_LOC, "failed to allocate workspace (%d)", workspace_size);
	}
	*cfg_and_state.output << " begins at " << (void*)net->workspace << std::endl;

	return 0;
}


int get_network_output_size(Darknet::Network & net)
{
	TAT(TATPARMS);

	int i;
	for (i = net.n-1; i > 0; --i)
	{
		if (net.layers[i].type != Darknet::ELayerType::COST)
		{
			break;
		}
	}

	return net.layers[i].outputs;
}


int get_network_input_size(Darknet::Network & net)
{
	TAT(TATPARMS);

	return net.layers[0].inputs;
}


Darknet::Image get_network_image_layer(Darknet::Network & net, int i)
{
	TAT(TATPARMS);

	Darknet::Layer & l = net.layers[i];
	if (l.out_w && l.out_h && l.out_c)
	{
		return Darknet::float_to_image(l.out_w, l.out_h, l.out_c, l.output);
	}
	Darknet::Image def = {0};
	return def;
}


Darknet::Image get_network_image(Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int i = net.n-1; i >= 0; --i)
	{
		Darknet::Image m = get_network_image_layer(net, i);
		if (m.h != 0)
		{
			return m;
		}
	}

	Darknet::Image def = {0};

	return def;
}


void visualize_network(Darknet::Network & net)
{
	TAT(TATPARMS);

	Darknet::Image * prev = 0;

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];

		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			std::string buffer = "Layer #" + std::to_string(i) + " (" + Darknet::to_string(static_cast<Darknet::ELayerType>(l.type)) + ")";
			prev = visualize_convolutional_layer(l, buffer.c_str(), prev);
		}
	}
}


// A version of network_predict that uses a pointer for the network
// struct to make the python binding work properly.
float *network_predict_ptr(DarknetNetworkPtr ptr, float * input)
{
	TAT(TATPARMS);

	// this is a "C" call

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);

	return network_predict(*net, input);
}


float *network_predict(Darknet::Network & net, float * input)
{
	TAT(TATPARMS);

	optimize_network_for_inference(net);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		return network_predict_gpu(net, input);
	}
#endif

	Darknet::NetworkState state = {0};
	state.net = net;
	state.index = 0;
	state.input = input;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	forward_network(net, state);
	float *out = get_network_output(net);

	return out;
}


int num_detections(Darknet::Network * net, float thresh)
{
	TAT(TATPARMS);

	int s = 0;
	for (int i = 0; i < net->n; ++i)
	{
		const Darknet::Layer & l = net->layers[i];
		if (l.type == Darknet::ELayerType::YOLO)
		{
			/// @todo V3 JAZZ:  this is where we spend all our time
			s += yolo_num_detections(l, thresh);
		}

		if (l.type == Darknet::ELayerType::YOLOV9)
		{
			s += yolov9_num_detections(net, l, thresh);
		}

		if (l.type == Darknet::ELayerType::GAUSSIAN_YOLO)
		{
			s += gaussian_yolo_num_detections(l, thresh);
		}

		if (l.type == Darknet::ELayerType::REGION)
		{
			s += l.w*l.h*l.n;
		}
	}

	return s;
}


/// Basically a wrapper for @ref yolo_num_detections_v3().  @returns the number of objects found in the current image
int num_detections_v3(Darknet::Network * net, float thresh, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	int detections = 0;

	for (int i = 0; i < net->n; ++i)
	{
		const Darknet::Layer & l = net->layers[i];
		if (l.type == Darknet::ELayerType::YOLO)
		{
			/// @todo V3 JAZZ:  this is where we spend all our time
			detections += yolo_num_detections_v3(net, i, thresh, cache);
		}
		else if (l.type == Darknet::ELayerType::YOLOV9)
		{
			detections += yolov9_num_detections_v3(net, i, thresh, cache);
		}

		/// @todo Is this still used in a modern .cfg file?  Should it be removed?
		else if (l.type == Darknet::ELayerType::GAUSSIAN_YOLO)
		{
			detections += gaussian_yolo_num_detections(l, thresh);
		}

		/// @todo Is this still used in a modern .cfg file?  Should it be removed?
		else if (l.type == Darknet::ELayerType::REGION)
		{
			detections += l.w * l.h * l.n;
		}
	}

	return detections;
}


int num_detections_batch(Darknet::Network * net, float thresh, int batch)
{
	TAT(TATPARMS);

	int s = 0;
	for (int i = 0; i < net->n; ++i)
	{
		const Darknet::Layer & l = net->layers[i];
		if (l.type == Darknet::ELayerType::YOLO)
		{
			s += yolo_num_detections_batch(l, thresh, batch);
		}
		else if (l.type == Darknet::ELayerType::YOLOV9)
		{
			s += yolov9_num_detections_batch(net, l, thresh, batch);
		}
		else if (l.type == Darknet::ELayerType::REGION)
		{
			s += l.w*l.h*l.n;
		}
	}

	return s;
}


DarknetDetection * make_network_boxes(DarknetNetworkPtr ptr, float thresh, int * num)
{
	TAT(TATPARMS);

	/// @see @ref make_network_boxes_batch()

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);

	// find the first layer that is one of these output types
	Darknet::Layer l;
	for (int i = 0; i < net->n; ++i)
	{
		Darknet::Layer & l_tmp = net->layers[i];
		if (l_tmp.type == Darknet::ELayerType::YOLO or
			l_tmp.type == Darknet::ELayerType::YOLOV9 or
			l_tmp.type == Darknet::ELayerType::GAUSSIAN_YOLO or
			l_tmp.type == Darknet::ELayerType::REGION or
			i == (net->n - 1))
		{
			l = l_tmp;
			break;
		}
	}

	/// @todo V3 JAZZ:  this is where we spend all our time
	const int nboxes = num_detections(net, thresh);
	if (num)
	{
		*num = nboxes;
	}

	DarknetDetection * dets = (DarknetDetection*)xcalloc(nboxes, sizeof(DarknetDetection));
	for (int i = 0; i < nboxes; ++i)
	{
		dets[i].prob = (float*)xcalloc(l.classes, sizeof(float));

		// tx,ty,tw,th uncertainty
		if (l.type == Darknet::ELayerType::GAUSSIAN_YOLO)
		{
			dets[i].uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
		}
		else
		{
			dets[i].uc = NULL;
		}

		if (l.coords > 4)
		{
			dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
		}
		else
		{
			dets[i].mask = NULL;
		}

		if (l.embedding_output)
		{
			dets[i].embeddings = (float*)xcalloc(l.embedding_size, sizeof(float));
		}
		else
		{
			dets[i].embeddings = NULL;
		}
		dets[i].embedding_size = l.embedding_size;
	}

	return dets;
}


Darknet::Detection * make_network_boxes_v3(Darknet::Network * net, const float thresh, int * num, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	// find a layer that is one of these types
	const Darknet::Layer & l = [&net]()
	{
		for (int i = 0; i < net->n; ++i)
		{
			/// @todo Is anything but YOLO still used as an output layer in a modern .cfg file?  Should these be removed?

			const Darknet::Layer & tmp = net->layers[i];
			if (tmp.type == Darknet::ELayerType::YOLO			or
				tmp.type == Darknet::ELayerType::YOLOV9			or
				tmp.type == Darknet::ELayerType::GAUSSIAN_YOLO	or
				tmp.type == Darknet::ELayerType::REGION			)
			{
				return tmp;
			}
		}

		// if nothing was found we'll use the last layer
		return net->layers[net->n - 1];
	}();

	/// @todo V3 JAZZ:  97% of this function is spent in this next line
	const int nboxes = num_detections_v3(net, thresh, cache);
	if (num)
	{
		*num = nboxes;
	}

	Darknet::Detection * dets = (Darknet::Detection*)xcalloc(nboxes, sizeof(Darknet::Detection));
	for (int i = 0; i < nboxes; ++i)
	{
		dets[i].prob = (float*)xcalloc(l.classes, sizeof(float));

		// tx,ty,tw,th uncertainty
		if (l.type == Darknet::ELayerType::GAUSSIAN_YOLO)
		{
			dets[i].uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
		}

		if (l.coords > 4)
		{
			dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
		}

		if (l.embedding_output)
		{
			dets[i].embeddings = (float*)xcalloc(l.embedding_size, sizeof(float));
		}

		dets[i].embedding_size = l.embedding_size;
	}

	return dets;
}


Darknet::Detection *make_network_boxes_batch(Darknet::Network * net, float thresh, int *num, int batch)
{
	TAT(TATPARMS);

	/// @see @ref make_network_boxes()

	Darknet::Layer l = net->layers[net->n - 1];
	for (int i = 0; i < net->n; ++i)
	{
		const Darknet::Layer & l_tmp = net->layers[i];
		if (l_tmp.type == Darknet::ELayerType::YOLO or
			l_tmp.type == Darknet::ELayerType::YOLOV9 or
			l_tmp.type == Darknet::ELayerType::GAUSSIAN_YOLO or
			l_tmp.type == Darknet::ELayerType::REGION)
		{
			l = l_tmp;
			break;
		}
	}

	const int nboxes = num_detections_batch(net, thresh, batch);
	assert(num != NULL);
	*num = nboxes;

	Darknet::Detection * dets = (Darknet::Detection*)calloc(nboxes, sizeof(Darknet::Detection));
	for (int i = 0; i < nboxes; ++i)
	{
		dets[i].prob = (float*)calloc(l.classes, sizeof(float));
		// tx,ty,tw,th uncertainty
		if (l.type == Darknet::ELayerType::GAUSSIAN_YOLO)
		{
			dets[i].uc = (float*)xcalloc(4, sizeof(float)); // Gaussian_YOLOv3
		}
		else
		{
			dets[i].uc = NULL;
		}

		if (l.coords > 4)
		{
			dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
		}
		else
		{
			dets[i].mask = NULL;
		}

		if (l.embedding_output)
		{
			dets[i].embeddings = (float*)xcalloc(l.embedding_size, sizeof(float));
		}
		else
		{
			dets[i].embeddings = NULL;
		}
		dets[i].embedding_size = l.embedding_size;
	}

	return dets;
}


void custom_get_region_detections(const Darknet::Layer & l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, Darknet::Detection *dets, int letter)
{
	TAT(TATPARMS);

	Darknet::Box * boxes = (Darknet::Box*)xcalloc(l.w * l.h * l.n, sizeof(Darknet::Box));
	float** probs = (float**)xcalloc(l.w * l.h * l.n, sizeof(float*));

	for (int j = 0; j < l.w*l.h*l.n; ++j)
	{
		probs[j] = (float*)xcalloc(l.classes, sizeof(float));
	}

	get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, map);
	for (int j = 0; j < l.w*l.h*l.n; ++j)
	{
		dets[j].classes = l.classes;
		dets[j].bbox = boxes[j];
		dets[j].objectness = 1;
		float highest_prob = 0;
		dets[j].best_class_idx = -1;
		for (int i = 0; i < l.classes; ++i)
		{
			if (probs[j][i] > highest_prob)
			{
				highest_prob = probs[j][i];
				dets[j].best_class_idx = i;
			}
			dets[j].prob[i] = probs[j][i];
		}
	}

	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);

	//correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
	correct_yolo_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}


void fill_network_boxes(Darknet::Network * net, int w, int h, float thresh, float hier, int *map, int relative, Darknet::Detection *dets, int letter)
{
	TAT(TATPARMS);

	int prev_classes = -1;
	for (int j = 0; j < net->n; ++j)
	{
		const Darknet::Layer & l = net->layers[j];
		switch (l.type)
		{
			case Darknet::ELayerType::YOLO:
			{
				/// @todo V3 JAZZ:  most of the time is spent in this function
				dets += get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);

				if (prev_classes < 0)
				{
					prev_classes = l.classes;
				}
				else if (prev_classes != l.classes)
				{
					darknet_fatal_error(DARKNET_LOC, "Different [yolo] layers have different number of classes (%d and %d)", prev_classes, l.classes);
				}
				break;
			}
			case Darknet::ELayerType::YOLOV9:
			{
				int count = get_yolov9_detections(net, l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
				dets += count;
				if (prev_classes < 0)
				{
					prev_classes = l.classes;
				}
				else if (prev_classes != l.classes)
				{
					darknet_fatal_error(DARKNET_LOC, "Different [yolov9] layers have different number of classes (%d and %d)", prev_classes, l.classes);
				}
				break;
			}
			case Darknet::ELayerType::GAUSSIAN_YOLO:
			{
				int count = get_gaussian_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
				dets += count;
				break;
			}
			case Darknet::ELayerType::REGION:
			{
				custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
				//get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
				dets += l.w*l.h*l.n;
				break;
			}
			default:
			{
				break;
			}
		}
	}
}


static inline void fill_network_boxes_v3(Darknet::Network * net, int w, int h, float thresh, float hier, int *map, int relative, Darknet::Detection *dets, int letter, Darknet::Output_Object_Cache & cache)
{
	TAT(TATPARMS);

	Darknet::Output_Object_Cache yolo_cache;
	Darknet::Output_Object_Cache yolov9_cache;
	for (const auto & oo : cache)
	{
		const auto type = net->layers[oo.layer_index].type;
		if (type == Darknet::ELayerType::YOLO)
		{
			yolo_cache.push_back(oo);
		}
		else if (type == Darknet::ELayerType::YOLOV9)
		{
			yolov9_cache.push_back(oo);
		}
		else
		{
			fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
			return;
		}
	}

	if (not yolo_cache.empty())
	{
		dets += get_yolo_detections_v3(net, w, h, net->w, net->h, thresh, map, relative, dets, letter, yolo_cache);
	}
	if (not yolov9_cache.empty())
	{
		dets += get_yolov9_detections_v3(net, w, h, net->w, net->h, thresh, map, relative, dets, letter, yolov9_cache);
	}
}


void fill_network_boxes_batch(Darknet::Network * net, int w, int h, float thresh, float hier, int *map, int relative, Darknet::Detection *dets, int letter, int batch)
{
	TAT(TATPARMS);

	int prev_classes = -1;
	for (int j = 0; j < net->n; ++j)
	{
		const Darknet::Layer & l = net->layers[j];
		if (l.type == Darknet::ELayerType::YOLO)
		{
			int count = get_yolo_detections_batch(l, w, h, net->w, net->h, thresh, map, relative, dets, letter, batch);
			dets += count;
			if (prev_classes < 0)
			{
				prev_classes = l.classes;
			}
			else if (prev_classes != l.classes)
			{
				darknet_fatal_error(DARKNET_LOC, "Different [yolo] layers have different number of classes = %d and %d - check your cfg-file!", prev_classes, l.classes);
			}
		}
		else if (l.type == Darknet::ELayerType::YOLOV9)
		{
			int count = get_yolov9_detections_batch(net, l, w, h, net->w, net->h, thresh, map, relative, dets, letter, batch);
			dets += count;
			if (prev_classes < 0)
			{
				prev_classes = l.classes;
			}
			else if (prev_classes != l.classes)
			{
				darknet_fatal_error(DARKNET_LOC, "Different [yolov9] layers have different number of classes = %d and %d - check your cfg-file!", prev_classes, l.classes);
			}
		}
		else if (l.type == Darknet::ELayerType::REGION)
		{
			custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
			dets += l.w*l.h*l.n;
		}
	}
}


DarknetDetection * get_network_boxes(DarknetNetworkPtr ptr, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
	TAT(TATPARMS);

	// this is a "C" call

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);

#if 0
	/* Prior to V3 Jazz, we'd call these 2 functions to create and fill in the bounding boxes.  The problem is both of
	 * them have to walk the entire output array looking for any object greater than the threshold.  As the network
	 * dimensions increase, the size of the output array increases as well.  So in V3, we introduced a "cache" to remember
	 * where the objects are located.  This means we only have to walk through the array once and remember all the
	 * locations of interest.
	 */

	detection *dets = make_network_boxes(net, thresh, num);
	fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
#else
	// With V3 Jazz, we now create a "cache" list to track objects in the output array.

	Darknet::Output_Object_Cache cache;
	cache.reserve(250); // reserve 250 spaces in the cache to start (most YOLO networks don't have 250 objects per image!)
	Darknet::Detection * dets = make_network_boxes_v3(net, thresh, num, cache);
	fill_network_boxes_v3(net, w, h, thresh, hier, map, relative, dets, letter, cache);
#endif

	return dets;
}


void free_detections(DarknetDetection * dets, int n)
{
	TAT(TATPARMS);

	// this is a "C" call

	for (int i = 0; i < n; ++i)
	{
		free(dets[i].prob);

		if (dets[i].uc)			free(dets[i].uc);
		if (dets[i].mask)		free(dets[i].mask);
		if (dets[i].embeddings)	free(dets[i].embeddings);
	}

	free(dets);
}


void free_batch_detections(det_num_pair *det_num_pairs, int n)
{
	TAT(TATPARMS);

	for(int i=0; i<n; ++i)
	{
		free_detections(det_num_pairs[i].dets, det_num_pairs[i].num);
	}
	free(det_num_pairs);
}


// JSON format:
//{
// "frame_id":8990,
// "objects":[
//  {"class_id":4, "name":"aeroplane", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.793070},
//  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.265497}
// ]
//},
char * Darknet::detection_to_json(Darknet::Detection *dets, int nboxes, int classes, const Darknet::VStr & names, long long int frame_id, char *filename)
{
	TAT(TATPARMS);

	const float thresh = 0.005; // function get_network_boxes() has already filtred dets by actual threshold

	char *send_buf = (char *)calloc(1024, sizeof(char));
	if (!send_buf)
	{
		return 0;
	}

	if (filename)
	{
		sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"filename\":\"%s\", \n \"objects\": [ \n", frame_id, filename);
	}
	else
	{
		sprintf(send_buf, "{\n \"frame_id\":%lld, \n \"objects\": [ \n", frame_id);
	}

	int class_id = -1;
	for (int i = 0; i < nboxes; ++i)
	{
		for (int j = 0; j < classes; ++j)
		{
			const bool show = (names[j].find("dont_show") != 0);
			if (dets[i].prob[j] > thresh && show)
			{
				if (class_id != -1)
				{
					strcat(send_buf, ", \n");
				}
				class_id = j;
				char *buf = (char *)calloc(2048, sizeof(char));
				if (!buf)
				{
					return 0;
				}

				sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, \"width\":%f, \"height\":%f}, \"confidence\":%f}",
					j, names[j].c_str(), dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]);

				int send_buf_len = strlen(send_buf);
				int buf_len = strlen(buf);
				int total_len = send_buf_len + buf_len + 100;
				send_buf = (char *)realloc(send_buf, total_len * sizeof(char));
				if (!send_buf)
				{
					if (buf)
					{
						free(buf);
					}
					return 0;// exit(-1);
				}
				strcat(send_buf, buf);
				free(buf);
			}
		}
	}

	strcat(send_buf, "\n ] \n}");

	return send_buf;
}


float * network_predict_image(DarknetNetworkPtr ptr, const DarknetImage im)
{
	TAT(TATPARMS);

	// this is a "C" call

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);

	if (net->batch != 1)
	{
		set_batch_network(net, 1);
	}

	float * p;
	if (im.w == net->w && im.h == net->h)
	{
		// Input image is the same size as our net, predict on that image
		p = network_predict(*net, im.data);
	}
	else
	{
		// need to resize image to the desired size for the net
		Darknet::Image imr = Darknet::resize_image(im, net->w, net->h);
		p = network_predict(*net, imr.data);
		Darknet::free_image(imr);
	}

	return p;
}


det_num_pair * network_predict_batch(Darknet::Network * net, Darknet::Image im, int batch_size, int w, int h, float thresh, float hier, int *map, int relative, int letter)
{
	TAT(TATPARMS);

	network_predict(*net, im.data);
	det_num_pair *pdets = (struct det_num_pair *)calloc(batch_size, sizeof(det_num_pair));

	int num = 0;
	for (int batch=0; batch < batch_size; batch++)
	{
		Darknet::Detection * dets = make_network_boxes_batch(net, thresh, &num, batch);
		fill_network_boxes_batch(net, w, h, thresh, hier, map, relative, dets, letter, batch);
		pdets[batch].num = num;
		pdets[batch].dets = dets;
	}

	return pdets;
}


float * network_predict_image_letterbox(DarknetNetworkPtr ptr, DarknetImage im)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);

	if (net->batch != 1)
	{
		set_batch_network(net, 1);
	}

	float * p;
	if (im.w == net->w && im.h == net->h)
	{
		// Input image is the same size as our net, predict on that image
		p = network_predict(*net, im.data);
	}
	else
	{
		// Need to resize image to the desired size for the net
		Darknet::Image imr = Darknet::letterbox_image(im, net->w, net->h);
		p = network_predict(*net, imr.data);
		Darknet::free_image(imr);
	}

	return p;
}


int network_width(Darknet::Network * net)
{
	TAT(TATPARMS);

	return net->w;
}


int network_height(Darknet::Network * net)
{
	TAT(TATPARMS);

	return net->h;
}


matrix network_predict_data_multi(Darknet::Network & net, data test, int n)
{
	TAT(TATPARMS);

	const int k = get_network_output_size(net);
	matrix pred = make_matrix(test.X.rows, k);
	float * X = (float*)xcalloc(net.batch * test.X.rows, sizeof(float));
	for (int i = 0; i < test.X.rows; i += net.batch)
	{
		for (int b = 0; b < net.batch; ++b)
		{
			if(i+b == test.X.rows)
			{
				break;
			}
			memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
		}

		for (int m = 0; m < n; ++m)
		{
			float *out = network_predict(net, X);
			for (int b = 0; b < net.batch; ++b)
			{
				if(i+b == test.X.rows)
				{
					break;
				}

				for (int j = 0; j < k; ++j)
				{
					pred.vals[i+b][j] += out[j+b*k]/n;
				}
			}
		}
	}

	free(X);

	return pred;
}


matrix network_predict_data(Darknet::Network & net, data test)
{
	TAT(TATPARMS);

	const int k = get_network_output_size(net);
	matrix pred = make_matrix(test.X.rows, k);
	float * X = (float*)xcalloc(net.batch * test.X.cols, sizeof(float));

	for (int i = 0; i < test.X.rows; i += net.batch)
	{
		for (int b = 0; b < net.batch; ++b)
		{
			if (i+b == test.X.rows)
			{
				break;
			}
			memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
		}

		float *out = network_predict(net, X);

		for (int b = 0; b < net.batch; ++b)
		{
			if (i+b == test.X.rows)
			{
				break;
			}

			for(int j = 0; j < k; ++j)
			{
				pred.vals[i+b][j] = out[j+b*k];
			}
		}
	}

	free(X);

	return pred;
}


void free_network_ptr(DarknetNetworkPtr ptr)
{
	TAT(TATPARMS);

	if (ptr)
	{
		Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
		free_network(*net);
	}

	return;
}


void free_network(Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int i = 0; i < net.n; ++i)
	{
		free_layer(net.layers[i]);
	}
	free(net.layers);
	free(net.seq_scales);
	free(net.scales);
	free(net.steps);
	free(net.seen);
	free(net.cuda_graph_ready);
	free(net.badlabels_reject_threshold);
	free(net.delta_rolling_max);
	free(net.delta_rolling_avg);
	free(net.delta_rolling_std);
	free(net.cur_iteration);
	free(net.total_bbox);
	free(net.rewritten_bbox);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(net.workspace);
	}
	else
	{
		free(net.workspace);
	}
	free_pinned_memory();
	if (net.input_state_gpu)
	{
		cuda_free(net.input_state_gpu);
	}
	if (net.input_pinned_cpu)
	{
		// CPU
		if (net.input_pinned_cpu_flag)
		{
			CHECK_CUDA(cudaFreeHost(net.input_pinned_cpu));
		}
		else
		{
			free(net.input_pinned_cpu);
		}
	}
	if (*net.input_gpu)			cuda_free(*net.input_gpu);
	if (*net.truth_gpu)			cuda_free(*net.truth_gpu);
	if (net.input_gpu)			free(net.input_gpu);
	if (net.truth_gpu)			free(net.truth_gpu);

	if (*net.input16_gpu)		cuda_free(*net.input16_gpu);
	if (*net.output16_gpu)		cuda_free(*net.output16_gpu);
	if (net.input16_gpu)		free(net.input16_gpu);
	if (net.output16_gpu)		free(net.output16_gpu);
	if (net.max_input16_size)	free(net.max_input16_size);
	if (net.max_output16_size)	free(net.max_output16_size);
#else
	free(net.workspace);
#endif

	if (net.details) // added in V3 (2024-08-06)
	{
		delete net.details;
		net.details = nullptr;
	}

	return;
}


static float lrelu(float src)
{
	TAT(TATPARMS);

	const float eps = 0.001;
	if (src > eps)
	{
		return src;
	}

	return eps;
}


void fuse_conv_batchnorm(Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int j = 0; j < net.n; ++j)
	{
		Darknet::Layer *l = &net.layers[j];

		if (l->type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			if (l->share_layer != NULL)
			{
				l->batch_normalize = 0;
			}

			if (l->batch_normalize)
			{
				for (int f = 0; f < l->n; ++f)
				{
					l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f] + .00001));

					double precomputed = l->scales[f] / (sqrt((double)l->rolling_variance[f] + .00001));

					const size_t filter_size = l->size*l->size*l->c / l->groups;
					for (int i = 0; i < filter_size; ++i)
					{
						int w_index = f*filter_size + i;

						l->weights[w_index] *= precomputed;
					}
				}

				free_convolutional_batchnorm(l);
				l->batch_normalize = 0;
#ifdef DARKNET_GPU
				if (cfg_and_state.gpu_index >= 0)
				{
					push_convolutional_layer(*l);
				}
#endif
			}
		}
		else if (l->type == Darknet::ELayerType::SHORTCUT && l->weights && l->weights_normalization)
		{
			if (l->nweights > 0)
			{
				//cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
				int i;
				for (i = 0; i < l->nweights; ++i)
				{
					*cfg_and_state.output << "w=" << l->weights[i] << ", ";
				}
				*cfg_and_state.output << "l->nweights=" << l->nweights << " j=" << j << std::endl;
			}

			const int layer_step = l->nweights / (l->n + 1);    // 1 or l.c or (l.c * l.h * l.w)

			for (int chan = 0; chan < layer_step; ++chan)
			{
				float sum = 1, max_val = -FLT_MAX;

				if (l->weights_normalization == SOFTMAX_NORMALIZATION)
				{
					for (int i = 0; i < (l->n + 1); ++i) {
						int w_index = chan + i * layer_step;
						float w = l->weights[w_index];
						if (max_val < w) max_val = w;
					}
				}

				const float eps = 0.0001;
				sum = eps;

				for (int i = 0; i < (l->n + 1); ++i)
				{
					int w_index = chan + i * layer_step;
					float w = l->weights[w_index];
					if (l->weights_normalization == RELU_NORMALIZATION) sum += lrelu(w);
					else if (l->weights_normalization == SOFTMAX_NORMALIZATION) sum += expf(w - max_val);
				}

				for (int i = 0; i < (l->n + 1); ++i)
				{
					int w_index = chan + i * layer_step;
					float w = l->weights[w_index];
					if (l->weights_normalization == RELU_NORMALIZATION) w = lrelu(w) / sum;
					else if (l->weights_normalization == SOFTMAX_NORMALIZATION) w = expf(w - max_val) / sum;
					l->weights[w_index] = w;
				}
			}

			l->weights_normalization = NO_NORMALIZATION;

#ifdef DARKNET_GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				push_shortcut_layer(*l);
			}
#endif
		}
	}
}


void forward_blank_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	return;
}


void calculate_binary_weights(DarknetNetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network & net = *reinterpret_cast<Darknet::Network *>(ptr);

	for (int j = 0; j < net.n; ++j)
	{
		Darknet::Layer *l = &net.layers[j];

		if (l->type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			if (l->xnor)
			{
				binary_align_weights(l);

				if (net.layers[j].use_bin_output)
				{
					l->activation = LINEAR;
				}

#ifdef DARKNET_GPU
				// fuse conv_xnor + shortcut -> conv_xnor
				if ((j + 1) < net.n && net.layers[j].type == Darknet::ELayerType::CONVOLUTIONAL)
				{
					Darknet::Layer *sc = &net.layers[j + 1];
					if (sc->type == Darknet::ELayerType::SHORTCUT && sc->w == sc->out_w && sc->h == sc->out_h && sc->c == sc->out_c)
					{
						l->bin_conv_shortcut_in_gpu = net.layers[net.layers[j + 1].index].output_gpu;
						l->bin_conv_shortcut_out_gpu = net.layers[j + 1].output_gpu;

						net.layers[j + 1].type = Darknet::ELayerType::BLANK;
						net.layers[j + 1].forward_gpu = forward_blank_layer;
					}
				}
#endif  // DARKNET_GPU
			}
		}
	}
}


void copy_cudnn_descriptors(const Darknet::Layer & src, Darknet::Layer *dst)
{
	TAT(TATPARMS);

#ifdef CUDNN
	dst->normTensorDesc = src.normTensorDesc;
	dst->normDstTensorDesc = src.normDstTensorDesc;
	dst->normDstTensorDescF16 = src.normDstTensorDescF16;

	dst->srcTensorDesc = src.srcTensorDesc;
	dst->dstTensorDesc = src.dstTensorDesc;

	dst->srcTensorDesc16 = src.srcTensorDesc16;
	dst->dstTensorDesc16 = src.dstTensorDesc16;
#endif // CUDNN
}


void copy_weights_net(const Darknet::Network & net_train, Darknet::Network * net_map)
{
	TAT(TATPARMS);

	for (int k = 0; k < net_train.n; ++k)
	{
		Darknet::Layer *l = &(net_train.layers[k]);
		Darknet::Layer tmp_layer;
		copy_cudnn_descriptors(net_map->layers[k], &tmp_layer);
		net_map->layers[k] = net_train.layers[k];
		copy_cudnn_descriptors(tmp_layer, &net_map->layers[k]);

		if (l->type == Darknet::ELayerType::CRNN)
		{
			Darknet::Layer tmp_input_layer, tmp_self_layer, tmp_output_layer;
			copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
			copy_cudnn_descriptors(*net_map->layers[k].self_layer, &tmp_self_layer);
			copy_cudnn_descriptors(*net_map->layers[k].output_layer, &tmp_output_layer);
			net_map->layers[k].input_layer = net_train.layers[k].input_layer;
			net_map->layers[k].self_layer = net_train.layers[k].self_layer;
			net_map->layers[k].output_layer = net_train.layers[k].output_layer;
			//net_map->layers[k].output_gpu = net_map->layers[k].output_layer->output_gpu;  // already copied out of if()

			copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
			copy_cudnn_descriptors(tmp_self_layer, net_map->layers[k].self_layer);
			copy_cudnn_descriptors(tmp_output_layer, net_map->layers[k].output_layer);
		}
		else if (l->input_layer) // for AntiAliasing
		{
			Darknet::Layer tmp_input_layer;
			copy_cudnn_descriptors(*net_map->layers[k].input_layer, &tmp_input_layer);
			net_map->layers[k].input_layer = net_train.layers[k].input_layer;
			copy_cudnn_descriptors(tmp_input_layer, net_map->layers[k].input_layer);
		}
		net_map->layers[k].batch = 1;
		net_map->layers[k].steps = 1;
		net_map->layers[k].train = 0;
	}
}


void free_network_recurrent_state(Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int k = 0; k < net.n; ++k)
	{
		if (net.layers[k].type == Darknet::ELayerType::CRNN)
		{
			free_state_crnn(net.layers[k]);
		}
	}
}


void restore_network_recurrent_state(Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int k = 0; k < net.n; ++k)
	{
		if (net.layers[k].type == Darknet::ELayerType::CRNN)
		{
			free_state_crnn(net.layers[k]);
		}
	}
}


int is_ema_initialized(const Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int i = 0; i < net.n; ++i)
	{
		const Darknet::Layer & l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			if (l.weights_ema)
			{
				for (int k = 0; k < l.nweights; ++k)
				{
					if (l.weights_ema[k] != 0)
					{
						return 1;
					}
				}
			}
		}
	}

	return 0;
}


void ema_update(Darknet::Network & net, float ema_alpha)
{
	TAT(TATPARMS);

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
#ifdef DARKNET_GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				pull_convolutional_layer(l);
			}
#endif

			if (l.weights_ema)
			{
				for (int k = 0; k < l.nweights; ++k)
				{
					l.weights_ema[k] = ema_alpha * l.weights_ema[k] + (1 - ema_alpha) * l.weights[k];
				}
			}

			for (int k = 0; k < l.n; ++k)
			{
				if (l.biases_ema)
				{
					l.biases_ema[k] = ema_alpha * l.biases_ema[k] + (1 - ema_alpha) * l.biases[k];
				}

				if (l.scales_ema)
				{
					l.scales_ema[k] = ema_alpha * l.scales_ema[k] + (1 - ema_alpha) * l.scales[k];
				}
			}
		}
	}
}


void ema_apply(Darknet::Network & net)
{
	TAT(TATPARMS);

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			if (l.weights_ema)
			{
				for (int k = 0; k < l.nweights; ++k)
				{
					l.weights[k] = l.weights_ema[k];
				}
			}

			for (int k = 0; k < l.n; ++k)
			{
				if (l.biases_ema)
				{
					l.biases[k] = l.biases_ema[k];
				}

				if (l.scales_ema)
				{
					l.scales[k] = l.scales_ema[k];
				}
			}

#ifdef DARKNET_GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				push_convolutional_layer(l);
			}
#endif
		}
	}
}


void reject_similar_weights(Darknet::Network & net, float sim_threshold)
{
	TAT(TATPARMS);

	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];

		if (i == 0)
		{
			continue;
		}

		if (net.n > i + 1) if (net.layers[i + 1].type == Darknet::ELayerType::YOLO) continue;
		if (net.n > i + 2) if (net.layers[i + 2].type == Darknet::ELayerType::YOLO) continue;
		if (net.n > i + 3) if (net.layers[i + 3].type == Darknet::ELayerType::YOLO) continue;

		if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.activation != LINEAR)
		{
#ifdef DARKNET_GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				pull_convolutional_layer(l);
			}
#endif
			float max_sim = -1000;
			int max_sim_index = 0;
			int max_sim_index2 = 0;
			int filter_size = l.size*l.size*l.c;
			for (int k = 0; k < l.n; ++k)
			{
				for (int j = k+1; j < l.n; ++j)
				{
					int w1 = k;
					int w2 = j;

					float sim = cosine_similarity(&l.weights[filter_size*w1], &l.weights[filter_size*w2], filter_size);
					if (sim > max_sim)
					{
						max_sim = sim;
						max_sim_index = w1;
						max_sim_index2 = w2;
					}
				}
			}

			*cfg_and_state.output
				<< "reject_similar_weights:"
				<< " i=" << i
				<< ", l.n=" << l.n
				<< ", w1=" << max_sim_index
				<< ", w2=" << max_sim_index2
				<< ", sim=" << max_sim
				<< ", thresh=" << sim_threshold
				<< std::endl;

			if (max_sim > sim_threshold)
			{
				*cfg_and_state.output << "rejecting..." << std::endl;
				float scale = sqrt(2. / (l.size*l.size*l.c / l.groups));

				/// @todo V5: would be nice if we could call rand_uniform_many() instead
				for (int k = 0; k < filter_size; ++k)
				{
					l.weights[max_sim_index*filter_size + k] = scale * rand_uniform(-1.0f, 1.0f);
				}
				if (l.biases) l.biases[max_sim_index] = 0.0f;
				if (l.scales) l.scales[max_sim_index] = 1.0f;
			}

#ifdef DARKNET_GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				push_convolutional_layer(l);
			}
#endif
		}
	}
}


cv::Mat Darknet::visualize_heatmap(const cv::Mat & heatmap, const cv::ColormapTypes colourmap)
{
	TAT(TATPARMS);

	if (heatmap.type() != CV_32FC1)
	{
		throw std::invalid_argument("heatmap should be of type 32FC1, not " + std::to_string(heatmap.type()));
	}

	double min_val = 0.0;
	double max_val = 0.0;
	cv::minMaxLoc(heatmap, &min_val, &max_val);

	// normalize the heatmap values
	cv::Mat mat = (heatmap - min_val) / (max_val - min_val);

	// convert the single-channel normalized heatmap to a colour image
	mat.convertTo(mat, CV_8UC1, 255.0); // mat contains normalized values, so multiply by 255 to get the full range of "bytes"
	cv::Mat colour;
	cv::applyColorMap(mat, colour, colourmap);

	return colour;
}
