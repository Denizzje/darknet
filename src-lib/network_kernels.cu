#include "darknet_internal.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	struct LayerForwardProfile
	{
		std::vector<cudaEvent_t> start_events;
		std::vector<cudaEvent_t> stop_events;
		std::vector<double> total_ms;
		std::vector<int> calls;
		std::vector<int> executed_this_frame;
		int layer_count = 0;
		int seen_frames = 0;
		int measured_frames = 0;
		int warmup_frames = 0;
		int target_frames = 0;
		bool printed = false;
	};

	static std::unordered_map<Darknet::Network *, LayerForwardProfile> layer_forward_profiles;

	static int get_env_int(const char * name, const int default_value)
	{
		const char * value = std::getenv(name);
		if (value == nullptr or value[0] == '\0')
		{
			return default_value;
		}
		const int parsed = std::atoi(value);
		return parsed > 0 ? parsed : default_value;
	}

	static bool gpu_layer_profile_enabled()
	{
		const char * value = std::getenv("DARKNET_PROFILE_LAYERS");
		return value != nullptr and value[0] != '\0' and std::string(value) != "0";
	}

	static bool env_flag_is_set(const char * name)
	{
		const char * value = std::getenv(name);
		return value != nullptr and value[0] != '\0' and std::string(value) != "0";
	}

	static bool cuda_graph_enabled(const Darknet::Network & net)
	{
		return net.use_cuda_graph == 1
			and not gpu_layer_profile_enabled()
			and not env_flag_is_set("DARKNET_DISABLE_INFERENCE_CUDA_GRAPH")
			and not env_flag_is_set("DARKNET_DISABLE_CUDA_GRAPH")
			and not env_flag_is_set("DARKNET_YOLOV9_PACK_INFERENCE_OUTPUT")
			and not env_flag_is_set("DARKNET_YOLOV9_PULL_IN_FORWARD");
	}

	static bool cuda_graph_verbose()
	{
		return cfg_and_state.is_verbose or env_flag_is_set("DARKNET_INFERENCE_OPTIMIZER_VERBOSE");
	}

	static std::mutex & cuda_graph_mutex_for(Darknet::Network & net)
	{
		static std::mutex registry_mutex;
		static std::map<void *, std::mutex> mutexes;

		std::lock_guard<std::mutex> registry_lock(registry_mutex);
		if (net.inference_cuda_graph_mutex == nullptr)
		{
			void * key = net.cuda_graph_ready ? static_cast<void *>(net.cuda_graph_ready) : static_cast<void *>(&net);
			net.inference_cuda_graph_mutex = static_cast<void *>(&mutexes[key]);
		}

		return *reinterpret_cast<std::mutex *>(net.inference_cuda_graph_mutex);
	}

	static LayerForwardProfile & get_layer_forward_profile(Darknet::Network & net)
	{
		LayerForwardProfile & profile = layer_forward_profiles[&net];
		if (profile.layer_count == net.n)
		{
			return profile;
		}

		for (cudaEvent_t event : profile.start_events)
		{
			if (event)
			{
				CHECK_CUDA(cudaEventDestroy(event));
			}
		}
		for (cudaEvent_t event : profile.stop_events)
		{
			if (event)
			{
				CHECK_CUDA(cudaEventDestroy(event));
			}
		}

		profile = {};
		profile.layer_count = net.n;
		profile.warmup_frames = get_env_int("DARKNET_PROFILE_LAYERS_WARMUP", 5);
		profile.target_frames = get_env_int("DARKNET_PROFILE_LAYERS_FRAMES", 30);
		profile.start_events.resize(net.n);
		profile.stop_events.resize(net.n);
		profile.total_ms.assign(net.n, 0.0);
		profile.calls.assign(net.n, 0);
		profile.executed_this_frame.reserve(net.n);

		for (int i = 0; i < net.n; ++i)
		{
			CHECK_CUDA(cudaEventCreate(&profile.start_events[i]));
			CHECK_CUDA(cudaEventCreate(&profile.stop_events[i]));
		}

		return profile;
	}

	static void print_layer_forward_profile(Darknet::Network & net, const LayerForwardProfile & profile)
	{
		const int top_count = get_env_int("DARKNET_PROFILE_LAYERS_TOP", 40);
		struct Row
		{
			int index;
			double total_ms;
			double avg_ms;
			int calls;
		};
		std::vector<Row> rows;
		rows.reserve(net.n);
		for (int i = 0; i < net.n; ++i)
		{
			if (profile.calls[i] <= 0)
			{
				continue;
			}
			rows.push_back(Row{i, profile.total_ms[i], profile.total_ms[i] / profile.calls[i], profile.calls[i]});
		}
		std::sort(rows.begin(), rows.end(), [](const Row & lhs, const Row & rhs)
		{
			return lhs.total_ms > rhs.total_ms;
		});

		struct TypeSummary
		{
			double total_ms = 0.0;
			int calls = 0;
		};
		std::map<Darknet::ELayerType, TypeSummary> by_type;
		for (const Row & row : rows)
		{
			TypeSummary & summary = by_type[net.layers[row.index].type];
			summary.total_ms += row.total_ms;
			summary.calls += row.calls;
		}
		std::vector<std::pair<Darknet::ELayerType, TypeSummary>> type_rows(by_type.begin(), by_type.end());
		std::sort(type_rows.begin(), type_rows.end(), [](const auto & lhs, const auto & rhs)
		{
			return lhs.second.total_ms > rhs.second.total_ms;
		});

		*cfg_and_state.output
			<< std::endl
			<< "CUDA layer profile over " << profile.measured_frames
			<< " inference forwards after " << profile.warmup_frames
			<< " warmup forwards:" << std::endl
			<< "Top layers by total forward time:" << std::endl;

		for (int rank = 0; rank < static_cast<int>(rows.size()) and rank < top_count; ++rank)
		{
			const Row & row = rows[rank];
			const Darknet::Layer & l = net.layers[row.index];
			*cfg_and_state.output
				<< "  #" << (rank + 1)
				<< " layer=" << row.index
				<< " type=" << Darknet::to_string(l.type)
				<< " avg_ms=" << row.avg_ms
				<< " total_ms=" << row.total_ms
				<< " calls=" << row.calls
				<< " out=" << l.out_w << "x" << l.out_h << "x" << l.out_c
				<< " filters=" << l.n
				<< " size=" << l.size
				<< " groups=" << l.groups
				<< std::endl;
		}

		*cfg_and_state.output << "Layer type summary:" << std::endl;
		for (const auto & [type, summary] : type_rows)
		{
			*cfg_and_state.output
				<< "  type=" << Darknet::to_string(type)
				<< " avg_ms=" << (summary.total_ms / summary.calls)
				<< " total_ms=" << summary.total_ms
				<< " calls=" << summary.calls
				<< std::endl;
		}
		*cfg_and_state.output << std::endl;
	}

	static void pull_yolov9_outputs_after_graph(Darknet::Network & net)
	{
		const bool force_full_pull = std::getenv("DARKNET_YOLOV9_PULL_AFTER_GRAPH") != nullptr;
		switch_stream(0);
		for (int layer_index = 0; layer_index < net.n; ++layer_index)
		{
			Darknet::Layer & l = net.layers[layer_index];
			if (l.type != Darknet::ELayerType::YOLOV9)
			{
				continue;
			}
			if (not force_full_pull)
			{
				l.inference_cpu_outputs_valid = 0;
				l.yolov9_compact_valid = 0;
				continue;
			}
			if (l.inference_cpu_outputs_valid)
			{
				continue;
			}
			const int branch = std::max(0, std::min(l.inference_branch, l.branch_count - 1));
			const int branch_offset = branch * l.n;
			for (int scale = 0; scale < l.n; ++scale)
			{
				const int slot = branch_offset + scale;
				if (slot < 0 or slot >= l.total)
				{
					continue;
				}
				Darknet::Layer & input = net.layers[l.input_layers[slot]];
				if (input.type == Darknet::ELayerType::ROUTE and input.inference_direct_yolov9_head and input.n == 2 and input.input_layers != nullptr)
				{
					Darknet::Layer & box = net.layers[input.input_layers[0]];
					Darknet::Layer & cls = net.layers[input.input_layers[1]];
					cuda_pull_array_async(box.output_gpu, box.output, box.batch * box.outputs);
					cuda_pull_array_async(cls.output_gpu, cls.output, cls.batch * cls.outputs);
				}
				else
				{
					cuda_pull_array_async(input.output_gpu, input.output, input.batch * input.outputs);
				}
			}
			CHECK_CUDA(cudaPeekAtLastError());
			CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
			l.inference_cpu_outputs_valid = 1;
		}
	}
}


typedef struct time_benchmark_layers
{
	float time;
	int layer_id;
	Darknet::ELayerType layer_type;
} time_benchmark_layers;


int time_comparator(const void *pa, const void *pb)
{
	TAT(TATPARMS);

	time_benchmark_layers a = *(time_benchmark_layers *)pa;
	time_benchmark_layers b = *(time_benchmark_layers *)pb;
	float diff = a.time - b.time;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void forward_network_gpu(Darknet::Network & net, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	static time_benchmark_layers *avg_time_per_layer = NULL;
	static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
	if (net.benchmark_layers)
	{
		if (!avg_time_per_layer)
		{
			avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
			sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
		}
		/// @todo in previous versions we did not CHECK_CUDA here -- was that intentional?
		CHECK_CUDA(cudaDeviceSynchronize()); // was this removed in CUDA 11.6+?
	}

	state.workspace = net.workspace;
	LayerForwardProfile * layer_profile = nullptr;
	bool profile_this_forward = false;
	if (not state.train and gpu_layer_profile_enabled())
	{
		layer_profile = &get_layer_forward_profile(net);
		if (not layer_profile->printed)
		{
			layer_profile->seen_frames += 1;
			profile_this_forward =
				layer_profile->seen_frames > layer_profile->warmup_frames and
				layer_profile->measured_frames < layer_profile->target_frames;
			if (profile_this_forward)
			{
				layer_profile->executed_this_frame.clear();
			}
		}
	}

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
				state.input = net.layers[alias_index].output_gpu + output_offset;
			}
			continue;
		}

		if (l.delta_gpu && state.train)
		{
			fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
		}

		if (profile_this_forward)
		{
			CHECK_CUDA(cudaEventRecord(layer_profile->start_events[i], get_cuda_stream()));
			layer_profile->executed_this_frame.push_back(i);
		}

		l.forward_gpu(l, state);

		if (profile_this_forward)
		{
			CHECK_CUDA(cudaEventRecord(layer_profile->stop_events[i], get_cuda_stream()));
		}

		if(net.wait_stream)
		{
			CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		}
		state.input = l.output_gpu;
	}

	if (profile_this_forward)
	{
		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		for (const int layer_index : layer_profile->executed_this_frame)
		{
			float elapsed_ms = 0.0f;
			CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, layer_profile->start_events[layer_index], layer_profile->stop_events[layer_index]));
			layer_profile->total_ms[layer_index] += elapsed_ms;
			layer_profile->calls[layer_index] += 1;
		}
		layer_profile->measured_frames += 1;
		if (layer_profile->measured_frames >= layer_profile->target_frames and not layer_profile->printed)
		{
			layer_profile->printed = true;
			print_layer_forward_profile(net, *layer_profile);
		}
	}

	if (net.benchmark_layers)
	{
		*cfg_and_state.output << std::endl << std::endl << "Sorted by time (forward):" << std::endl;

		/// @todo replace qsort() low priority
		qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);

		for (int i = 0; i < net.n; ++i)
		{
			*cfg_and_state.output
				<< i
				<< " - fw-sort-layer " << sorted_avg_time_per_layer[i].layer_id
				<< " - type: " << static_cast<int>(sorted_avg_time_per_layer[i].layer_type)
				<< " - avg_time " << sorted_avg_time_per_layer[i].time << " ms"
				<< std::endl;
		}
	}
}

void backward_network_gpu(Darknet::Network & net, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	static time_benchmark_layers *avg_time_per_layer = NULL;
	static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
	if (net.benchmark_layers)
	{
		if (!avg_time_per_layer)
		{
			avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
			sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
		}
		CHECK_CUDA(cudaDeviceSynchronize());
	}

	state.workspace = net.workspace;
	int i;
	float * original_input = state.input;
	float * original_delta = state.delta;
	for(i = net.n-1; i >= 0; --i)
	{
		state.index = i;
		Darknet::Layer & l = net.layers[i];
		if (l.stopbackward == 1)
		{
			break;
		}

		if (l.stopbackward > get_current_iteration(net))
		{
			break;
		}

		if (i == 0)
		{
			state.input = original_input;
			state.delta = original_delta;
		}
		else
		{
			const Darknet::Layer & prev = net.layers[i-1];
			state.input = prev.output_gpu;
			state.delta = prev.delta_gpu;
			if (net.optimized_memory && !prev.keep_delta_gpu)
			{
				state.delta = net.state_delta_gpu;
			}
		}

		if (l.onlyforward)
		{
			continue;
		}

		l.backward_gpu(l, state);

		if (i != 0)
		{
			Darknet::Layer & prev = net.layers[i - 1];
			if (net.optimized_memory && state.delta && !prev.keep_delta_gpu)
			{
				if (prev.delta_gpu != state.delta)
				{
					simple_copy_ongpu(prev.outputs*prev.batch, state.delta, prev.delta_gpu);
				}
				fill_ongpu(prev.outputs*prev.batch, 0, net.state_delta_gpu, 1);
			}
		}
	}

	if (net.adversarial && net.attention)
	{
		int img_size = net.w * net.h * net.c;
		float *original_input_cpu = (float *)xcalloc(img_size, sizeof(float));
		float *original_delta_cpu = (float *)xcalloc(img_size, sizeof(float));
		cuda_pull_array(original_input, original_input_cpu, img_size);
		cuda_pull_array(original_delta, original_delta_cpu, img_size);

		Darknet::Image attention_img = Darknet::make_attention_image(img_size, original_delta_cpu, original_input_cpu, net.w, net.h, net.c, 0.7);
		Darknet::show_image(attention_img, "attention_img");
		cv::resizeWindow("attention_img", 500, 500);

		Darknet::free_image(attention_img);

		Darknet::Image attention_mask_img = Darknet::make_attention_image(img_size, original_delta_cpu, original_delta_cpu, net.w, net.h, net.c, 1.0);
		Darknet::show_image(attention_mask_img, "attention_mask_img");
		cv::resizeWindow("attention_mask_img", 500, 500);

		Darknet::free_image(attention_mask_img);

		free(original_input_cpu);
		free(original_delta_cpu);
	}

	if (net.adversarial)
	{
		int x_size = get_network_input_size(net) * net.batch;
		*cfg_and_state.output
			<< "x_size=" << x_size
			<< ", original_delta=" << original_delta
			<< ", original_input=" << original_input
			<< ", net.learning_rate=" << net.learning_rate
			<< std::endl;
		axpy_ongpu(x_size, net.learning_rate, original_delta, 1, original_input, 1);
		constrain_min_max_ongpu(x_size, 0, 1, original_input, 1);
	}

	if (net.benchmark_layers)
	{
		*cfg_and_state.output << std::endl << std::endl << "Sorted by time (backward):" << std::endl;

		/// @todo replace qsort() unknown priority
		qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);

		for (i = 0; i < net.n; ++i)
		{
			*cfg_and_state.output
				<< i
				<< " - bw-sort-layer " << sorted_avg_time_per_layer[i].layer_id
				<< " - type: " << static_cast<int>(sorted_avg_time_per_layer[i].layer_type)
				<< " - avg_time " << sorted_avg_time_per_layer[i].time << " ms"
				<< std::endl;
		}
	}
}

void update_network_gpu(Darknet::Network & net)
{
	TAT(TATPARMS);

	cuda_set_device(net.gpu_index);
	const int iteration_num = (*net.seen) / (net.batch * net.subdivisions);

	int update_batch = net.batch*net.subdivisions * get_sequence_value(net);

	float rate = get_current_rate(net);
	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.train == 0)
		{
			continue;
		}
		l.t = get_current_batch(net);
		if (iteration_num > (net.max_batches * 1 / 2))
		{
			l.deform = 0;
		}
		if (l.burnin_update && (l.burnin_update*net.burn_in > iteration_num))
		{
			continue;
		}
		if (l.train_only_bn)
		{
			continue;
		}

		if (l.update_gpu && l.dont_update < iteration_num)
		{
			l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
		}
	}
}

void forward_backward_network_gpu(Darknet::Network & net, float *x, float *y)
{
	TAT(TATPARMS);

	Darknet::NetworkState state;
	state.index = 0;
	state.net = net;
	int x_size = get_network_input_size(net)*net.batch;
	int y_size = get_network_output_size(net)*net.batch;
	if (net.layers[net.n-1].truths)
	{
		y_size = net.layers[net.n-1].truths*net.batch;
	}
	if (!*net.input_gpu)
	{
		*net.input_gpu = cuda_make_array(x, x_size);
		*net.truth_gpu = cuda_make_array(y, y_size);
	}
	else
	{
		cuda_push_array(*net.input_gpu, x, x_size);
		cuda_push_array(*net.truth_gpu, y, y_size);
	}
	state.input = *net.input_gpu;
	state.delta = 0;
	if (net.adversarial)
	{
		state.delta = cuda_make_array(NULL, x_size);
	}
	state.truth = *net.truth_gpu;
	state.train = 1;
#if defined(CUDNN_HALF) && defined(CUDNN)
	int i;
	for (i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (net.cudnn_half)
		{
			if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.weights_gpu && l.weights_gpu16)
			{
				assert((l.nweights) > 0);
				cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
			}
			else if (l.type == Darknet::ELayerType::CRNN && l.input_layer->weights_gpu && l.input_layer->weights_gpu16)
			{
				assert((l.input_layer->c*l.input_layer->n*l.input_layer->size*l.input_layer->size) > 0);
				cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
				cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
				cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
			}
		}
	}
#endif
	forward_network_gpu(net, state);
	//cudaStreamSynchronize(get_cuda_stream());
	backward_network_gpu(net, state);

	if (net.adversarial)
	{
		cuda_free(state.delta);
		cuda_pull_array(*net.input_gpu, x, x_size);
	}
}

float train_network_datum_gpu(Darknet::Network & net, float *x, float *y)
{
	TAT(TATPARMS);

	*net.seen += net.batch;
	if (net.adversarial_lr && rand_bool() && get_current_iteration(net) > net.burn_in)
	{
		net.adversarial = 1;
		float lr_old = net.learning_rate;
		float scale = (get_current_iteration(net) / ((float)net.max_batches));
		//scale = sin(scale * M_PI);
		net.learning_rate = net.adversarial_lr * scale;
		int y_size = get_network_output_size(net)*net.batch;
		if (net.layers[net.n - 1].truths)
		{
			y_size = net.layers[net.n - 1].truths*net.batch;
		}
		float *truth_cpu = (float *)xcalloc(y_size, sizeof(float));

		const int img_size = net.w*net.h*net.c;
		float *old_input = (float *)xcalloc(img_size*net.batch, sizeof(float));
		memcpy(old_input, x, img_size*net.batch * sizeof(float));

		*cfg_and_state.output << std::endl << "adversarial training, adversarial_lr=" << net.adversarial_lr * scale << std::endl;

		forward_backward_network_gpu(net, x, truth_cpu);

		int b;
		for (b = 0; b < net.batch; ++b)
		{
			if (b % 2 == 1 && net.contrastive)
			{
				memcpy(x + img_size*b, old_input + img_size*b, img_size * sizeof(float));
			}
		}

		Darknet::Image im;
		im.w = net.w;
		im.h = net.h;
		im.c = net.c;
		im.data = x;
		Darknet::show_image(im, "adversarial data augmentation");
		cv::resizeWindow("adversarial data augmentation", 500, 500);
		cv::waitKey(1);

		free(old_input);
		free(truth_cpu);
		net.learning_rate = lr_old;
		net.adversarial = 0;
	}
	forward_backward_network_gpu(net, x, y);
	float error = get_network_cost(net);

	return error;
}


void pull_updates(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		if(l.scale_updates)
		{
			cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
	}
}

void push_updates(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
	}
}

void update_layer(Darknet::Layer & l, Darknet::Network net)
{
	TAT(TATPARMS);

	int update_batch = net.batch*net.subdivisions;
	float rate = get_current_rate(net);
	l.t = get_current_batch(net);
	if(l.update_gpu)
	{
		l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
	}
}

void merge_weights(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
		axpy_cpu(l.nweights, 1, l.weights, 1, base.weights, 1);
		if (l.scales)
		{
			axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
		axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
	}
}

void scale_weights(Darknet::Layer & l, float s)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		scal_cpu(l.n, s, l.biases, 1);
		scal_cpu(l.nweights, s, l.weights, 1);
		if (l.scales)
		{
			scal_cpu(l.n, s, l.scales, 1);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		scal_cpu(l.outputs, s, l.biases, 1);
		scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
	}
}


void pull_weights(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_pull_array(l.biases_gpu, l.biases, l.n);
		cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
		if (l.scales)
		{
			cuda_pull_array(l.scales_gpu, l.scales, l.n);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
		cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
	}
}

void push_weights(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if(l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.biases_gpu, l.biases, l.n);
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
		if(l.scales)
		{
			cuda_push_array(l.scales_gpu, l.scales, l.n);
		}
	}
	else if(l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.biases_gpu, l.biases, l.outputs);
		cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
	}
}

void distribute_weights(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if(l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.biases_gpu, base.biases, l.n);
		cuda_push_array(l.weights_gpu, base.weights, l.nweights);
		if(base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
	}
	else if(l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.biases_gpu, base.biases, l.outputs);
		cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
	}
}


void merge_updates(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL) {
		axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
		axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
		if (l.scale_updates) {
			axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
		}
	} else if(l.type == Darknet::ELayerType::CONNECTED) {
		axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
		axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
	}
}

void distribute_updates(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if(l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
		cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
		if(base.scale_updates)
		{
			cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
		cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
	}
}

void sync_layer(Darknet::Network * nets, int n, int j)
{
	TAT(TATPARMS);

	Darknet::Network net = nets[0];
	Darknet::Layer & base = net.layers[j];
	cuda_set_device(net.gpu_index);
	pull_weights(base);

	for (int i = 1; i < n; ++i)
	{
		cuda_set_device(nets[i].gpu_index);
		Darknet::Layer & l = nets[i].layers[j];
		pull_weights(l);
		merge_weights(l, base);
	}

	scale_weights(base, 1./n);

	for (int i = 0; i < n; ++i)
	{
		cuda_set_device(nets[i].gpu_index);
		Darknet::Layer & l = nets[i].layers[j];
		distribute_weights(l, base);
	}
}


void sync_nets(Darknet::Network * nets, int n, int interval)
{
	TAT(TATPARMS);

	int layers = nets[0].n;

	std::vector<std::thread> threads;
	threads.reserve(layers);

	*nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
	for (int j = 0; j < n; ++j)
	{
		*nets[j].seen = *nets[0].seen;
	}

	for (int j = 0; j < layers; ++j)
	{
		threads.emplace_back(
				[nets,n,j]()
				{
					sync_layer(nets, n, j);
				});
	}

	for (auto & t : threads)
	{
		t.join();
	}

	return;
}

float train_networks(Darknet::Network * nets, int n, data d, int interval)
{
	TAT(TATPARMS);

	// IMPORTANT:  If we get here, we already know that n > 1!  This is only called when we have multiple GPUs.
	// There is another similar function called train_network() for single GPU (note singular name!)

#ifdef _DEBUG
	int batch = nets[0].batch;
	int subdivisions = nets[0].subdivisions;
	assert(batch * subdivisions * n == d.X.rows);
#endif

	// "errors"?  This is "loss", right?  We're adding up the loss from training a batch on each GPU?
	float * errors = (float*) calloc(n, sizeof(float));

	std::vector<std::thread> threads;
	threads.reserve(n);
	std::vector<data> p(n);

	for (int i = 0; i < n; ++i)
	{
		 p[i] = get_data_part(d, i, n);

		threads.emplace_back(
			[](Darknet::Network & net, data &d2, float * err)
			{
				TAT(TATPARMS);

				cuda_set_device(net.gpu_index);
				*err = train_network(net, d2); // note this is the "singular" train function (e.g., for a single GPU)
			},
			std::ref(nets[i]), std::ref(p[i]), errors + i);
	}

	float sum = 0.0f;
	for (int i = 0; i < n; ++i)
	{
		threads[i].join();
		sum += errors[i];
	}
	free(errors);

	//cudaDeviceSynchronize();
	*nets[0].cur_iteration += (n - 1);
	*nets[0].seen = nets[0].batch * nets[0].subdivisions * get_current_iteration(nets[0]); // remove this line, when you will save to weights-file both: seen & cur_iteration
	if (get_current_iteration(nets[0]) % interval == 0)
	{
		if (cfg_and_state.is_verbose)
		{
			*cfg_and_state.output << "Syncing..." << std::flush;
		}
		sync_nets(nets, n, interval);
		if (cfg_and_state.is_verbose)
		{
			*cfg_and_state.output << "done!" << std::endl;
		}
	}

	//cudaDeviceSynchronize();
	return sum / n;
}

float *get_network_output_layer_gpu(Darknet::Network & net, int i)
{
	TAT(TATPARMS);

	const int output_offset = resolve_inference_layer_output_offset(net, i);
	i = resolve_inference_layer_index(net, i);
	Darknet::Layer & l = net.layers[i];
	const bool graph_output_current =
		net.inference_cuda_graph_enabled != 0 and
		net.cuda_graph_ready != nullptr and
		*net.cuda_graph_ready != 0;
	if (l.type != Darknet::ELayerType::REGION && l.type != Darknet::ELayerType::YOLO && l.type != Darknet::ELayerType::YOLOV9 && not graph_output_current)
	{
		cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
	}

	return l.output + output_offset;
}

float *get_network_output_gpu(Darknet::Network & net)
{
	TAT(TATPARMS);

	int i;
	for (i = net.n - 1; i > 0; --i)
	{
		if (net.layers[i].type != Darknet::ELayerType::COST)
		{
			break;
		}
	}

	return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(Darknet::Network & net, float *input)
{
	TAT(TATPARMS);

	optimize_network_for_inference(net);
	const bool use_graph = cuda_graph_enabled(net);
	net.inference_cuda_graph_enabled = use_graph ? 1 : 0;
	std::unique_lock<std::mutex> graph_lock;
	if (use_graph)
	{
		graph_lock = std::unique_lock<std::mutex>(cuda_graph_mutex_for(net));
		const bool graph_was_ready = net.cuda_graph_ready != nullptr and *net.cuda_graph_ready != 0;
		net.inference_cuda_graph_captured = graph_was_ready ? 1 : 0;
	}
	else
	{
		const bool graph_was_ready = net.cuda_graph_ready != nullptr and *net.cuda_graph_ready != 0;
		net.inference_cuda_graph_captured = graph_was_ready ? 1 : 0;
	}

	if (net.gpu_index != cuda_get_device())
	{
		cuda_set_device(net.gpu_index);
	}
	int size = get_network_input_size(net) * net.batch;
	Darknet::NetworkState state;
	state.index = 0;
	state.net = net;
	//state.input = cuda_make_array(input, size);   // memory will be allocated in the parse_network_cfg_custom()
	state.input = net.input_state_gpu;
	memcpy(net.input_pinned_cpu, input, size * sizeof(float));
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	for (int i = 0; i < net.n; ++i)
	{
		if (net.layers[i].type == Darknet::ELayerType::YOLOV9)
		{
			net.layers[i].inference_cpu_outputs_valid = 0;
			net.layers[i].yolov9_compact_valid = 0;
		}
	}

	if (use_graph)
	{
		switch_stream(0);
	}
	cuda_push_array(state.input, net.input_pinned_cpu, size);

	if (use_graph)
	{
		cudaGraphExec_t instance = reinterpret_cast<cudaGraphExec_t>(net.cuda_graph_exec);
		if ((*net.cuda_graph_ready) == 0 or instance == nullptr)
		{
			for (int i = 0; i < 16; ++i)
			{
				switch_stream(i);
			}

			cudaStream_t stream0 = switch_stream(0);

			// cuDNN/cuBLAS may lazily initialize kernels and workspaces on the first pass.
			// Run that pass outside capture, then capture the steady-state inference graph.
			CHECK_CUDA(cudaDeviceSynchronize());
			forward_network_gpu(net, state);
			CHECK_CUDA(cudaStreamSynchronize(stream0));

			for (int i = 0; i < net.n; ++i)
			{
				if (net.layers[i].type == Darknet::ELayerType::YOLOV9)
				{
					net.layers[i].inference_cpu_outputs_valid = 0;
					net.layers[i].yolov9_compact_valid = 0;
				}
			}

			if (cuda_graph_verbose())
			{
				*cfg_and_state.output << "Try to capture graph..." << std::endl;
			}
			cudaGraph_t graph = nullptr;
			CHECK_CUDA(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));

			forward_network_gpu(net, state);

			CHECK_CUDA(cudaStreamEndCapture(stream0, &graph));
			CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
			net.cuda_graph = reinterpret_cast<void *>(graph);
			net.cuda_graph_exec = reinterpret_cast<void *>(instance);
			(*net.cuda_graph_ready) = 1;
			net.inference_cuda_graph_captured = 1;
			if (cuda_graph_verbose())
			{
				*cfg_and_state.output << "Graph is captured..." << std::endl;
			}
			CHECK_CUDA(cudaDeviceSynchronize());
		}
		else
		{
			cudaStream_t stream0 = switch_stream(0);
			CHECK_CUDA(cudaGraphLaunch(instance, stream0));
			CHECK_CUDA(cudaStreamSynchronize(stream0));
			net.inference_cuda_graph_launches += 1;
		}
	}
	else
	{
		forward_network_gpu(net, state);
		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
	}

	float *out = get_network_output_gpu(net);
	if (use_graph)
	{
		pull_yolov9_outputs_after_graph(net);
	}
	reset_wait_stream_events();
	//cuda_free(state.input);   // will be freed in the free_network()
	return out;
}
