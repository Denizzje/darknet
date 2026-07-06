#include "data.hpp"
#include "darknet_internal.hpp"

#include <array>
#include <fstream>
#include <sstream>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	/** The permanent data-loading (image, bboxes) threads started by @ref Darknet::run_image_loading_control_thread().
	 *
	 * @since 2024-04-03
	 */
	static Darknet::VThreads data_loading_threads;


	/** Flag used by the image data loading threads to determine if they need to exit.
	 *
	 * @since 2024-04-02
	 */
	static std::atomic<bool> image_data_loading_threads_must_exit = false;


	/** Flags to indicate to individual data loading threads what they should do.  @p 0 is stop, and @p 1 is go.
	 * These flags are normally @p 0 and then are set to @p 1 by @ref run_image_loading_control_thread().
	 *
	 * (Was @p std::vector<bool> but that individual bit handling, and we only have a few threads.)
	 *
	 * @since 2024-04-10
	 */
	static std::vector<int> data_loading_per_thread_flag;


	/// @{ @todo: delete these once the code is cleaned up
	static const std::chrono::milliseconds thread_wait_ms(5); ///< @todo DELETE THIS! :(
	static load_args * args_swap = NULL; ///< @todo I wish I better understood how/why this exists...
	static std::mutex args_swap_mutex; // used to protect access to args_swap
	/// @}


	static inline data concat_datas(data *d, int n)
	{
		TAT(TATPARMS);

		data out = {0};
		for (int i = 0; i < n; ++i)
		{
			data newdata = concat_data(d[i], out);
			Darknet::free_data(out);
			out = newdata;
		}

		return out;
	}

	static inline std::mt19937 & get_rnd_engine()
	{
		TAT(TATPARMS);

		// we must have 1 per thread of these (use random_device to seed the engine)
		static thread_local std::mt19937 rnd_engine(std::random_device{}());

		return rnd_engine;
	}


	struct Yolov9Label
	{
		int cls = 0;
		int track_id = 0;
		cv::Rect2f box;
		std::vector<cv::Point2f> segment;
	};


	struct Yolov9Sample
	{
		cv::Mat image;
		std::vector<Yolov9Label> labels;
	};


	struct Yolov9Letterbox
	{
		cv::Mat image;
		float ratio = 1.0f;
		float padw = 0.0f;
		float padh = 0.0f;
	};


	static inline int python_round(const double value)
	{
		return static_cast<int>(std::nearbyint(value));
	}


	static inline void clip_label(Yolov9Label & label, const float width, const float height)
	{
		const float x1 = std::clamp(label.box.x, 0.0f, width);
		const float y1 = std::clamp(label.box.y, 0.0f, height);
		const float x2 = std::clamp(label.box.x + label.box.width, 0.0f, width);
		const float y2 = std::clamp(label.box.y + label.box.height, 0.0f, height);
		label.box = cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));

		for (auto & point : label.segment)
		{
			point.x = std::clamp(point.x, 0.0f, width);
			point.y = std::clamp(point.y, 0.0f, height);
		}
	}


	static inline float label_area(const Yolov9Label & label)
	{
		return std::max(0.0f, label.box.width) * std::max(0.0f, label.box.height);
	}


	static std::vector<Yolov9Label> read_yolov9_labels(const char * image_path, const int classes, const int image_w, const int image_h)
	{
		TAT(TATPARMS);

		char label_path[4096];
		replace_image_to_label(image_path, label_path);

		std::ifstream stream(label_path);
		if (not stream.good())
		{
			darknet_fatal_error(DARKNET_LOC, "failed to open annotation file \"%s\"", label_path);
		}

		std::vector<Yolov9Label> labels;
		std::string line;
		int line_counter = 0;
		const int img_hash = (custom_hash(label_path) % 4000) * 4000;

		while (std::getline(stream, line))
		{
			++line_counter;
			if (line.empty())
			{
				continue;
			}

			std::istringstream values_stream(line);
			std::vector<float> values;
			float value = 0.0f;
			while (values_stream >> value)
			{
				values.push_back(value);
			}

			if (values.empty())
			{
				continue;
			}

			if (values.size() != 5 and (values.size() < 7 or values.size() % 2 == 0))
			{
				darknet_fatal_error(DARKNET_LOC, "invalid YOLO label in \"%s\" on line #%d", label_path, line_counter);
			}

			Yolov9Label label;
			label.cls = static_cast<int>(values[0]);
			label.track_id = img_hash + line_counter;
			if (label.cls < 0 or label.cls >= classes)
			{
				darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d in %s on line #%d", label.cls, label_path, line_counter);
			}

			if (values.size() == 5)
			{
				const float cx = values[1] * image_w;
				const float cy = values[2] * image_h;
				const float bw = values[3] * image_w;
				const float bh = values[4] * image_h;
				label.box = cv::Rect2f(cx - bw / 2.0f, cy - bh / 2.0f, bw, bh);
			}
			else
			{
				float x_min = static_cast<float>(image_w);
				float y_min = static_cast<float>(image_h);
				float x_max = 0.0f;
				float y_max = 0.0f;
				for (size_t idx = 1; idx + 1 < values.size(); idx += 2)
				{
					cv::Point2f point(values[idx] * image_w, values[idx + 1] * image_h);
					label.segment.push_back(point);
					x_min = std::min(x_min, point.x);
					y_min = std::min(y_min, point.y);
					x_max = std::max(x_max, point.x);
					y_max = std::max(y_max, point.y);
				}
				label.box = cv::Rect2f(x_min, y_min, x_max - x_min, y_max - y_min);
			}

			clip_label(label, static_cast<float>(image_w), static_cast<float>(image_h));
			if (label_area(label) > 0.0f)
			{
				labels.push_back(label);
			}
		}

		return labels;
	}


	static Yolov9Sample load_yolov9_resized_sample(const char * filename, const load_args & args)
	{
		TAT(TATPARMS);

		Yolov9Sample sample;
		sample.image = load_rgb_mat_image(filename, args.c);

		const int h0 = sample.image.rows;
		const int w0 = sample.image.cols;
		const int target = std::max(args.w, args.h);
		const float ratio = static_cast<float>(target) / static_cast<float>(std::max(h0, w0));

		if (ratio != 1.0f)
		{
			cv::resize(sample.image, sample.image, cv::Size(static_cast<int>(w0 * ratio), static_cast<int>(h0 * ratio)), 0, 0, cv::INTER_LINEAR);
		}

		sample.labels = read_yolov9_labels(filename, args.classes, sample.image.cols, sample.image.rows);
		return sample;
	}


	static Yolov9Letterbox yolov9_letterbox_rgb(const cv::Mat & image, const int target_w, const int target_h)
	{
		TAT(TATPARMS);

		Yolov9Letterbox result;
		const int src_h = image.rows;
		const int src_w = image.cols;
		const float ratio = std::min(static_cast<float>(target_h) / static_cast<float>(src_h), static_cast<float>(target_w) / static_cast<float>(src_w));
		const cv::Size new_unpad(python_round(src_w * ratio), python_round(src_h * ratio));
		float dw = static_cast<float>(target_w - new_unpad.width);
		float dh = static_cast<float>(target_h - new_unpad.height);
		dw /= 2.0f;
		dh /= 2.0f;

		cv::Mat resized;
		if (src_w != new_unpad.width or src_h != new_unpad.height)
		{
			cv::resize(image, resized, new_unpad, 0, 0, cv::INTER_LINEAR);
		}
		else
		{
			resized = image.clone();
		}

		const int top = python_round(dh - 0.1);
		const int bottom = python_round(dh + 0.1);
		const int left = python_round(dw - 0.1);
		const int right = python_round(dw + 0.1);
		cv::copyMakeBorder(resized, result.image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
		result.ratio = ratio;
		result.padw = dw;
		result.padh = dh;
		return result;
	}


	static void apply_letterbox_to_labels(std::vector<Yolov9Label> & labels, const Yolov9Letterbox & letterbox)
	{
		TAT(TATPARMS);

		for (auto & label : labels)
		{
			label.box.x = label.box.x * letterbox.ratio + letterbox.padw;
			label.box.y = label.box.y * letterbox.ratio + letterbox.padh;
			label.box.width *= letterbox.ratio;
			label.box.height *= letterbox.ratio;
			for (auto & point : label.segment)
			{
				point.x = point.x * letterbox.ratio + letterbox.padw;
				point.y = point.y * letterbox.ratio + letterbox.padh;
			}
		}
	}


	static cv::Point2f transform_point(const cv::Point2f & point, const cv::Matx33d & matrix, const bool perspective)
	{
		const double x = matrix(0, 0) * point.x + matrix(0, 1) * point.y + matrix(0, 2);
		const double y = matrix(1, 0) * point.x + matrix(1, 1) * point.y + matrix(1, 2);
		const double z = matrix(2, 0) * point.x + matrix(2, 1) * point.y + matrix(2, 2);
		if (perspective and z != 0.0)
		{
			return cv::Point2f(static_cast<float>(x / z), static_cast<float>(y / z));
		}
		return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
	}


	static cv::Rect2f rect_from_points(const std::vector<cv::Point2f> & points)
	{
		float x_min = std::numeric_limits<float>::max();
		float y_min = std::numeric_limits<float>::max();
		float x_max = std::numeric_limits<float>::lowest();
		float y_max = std::numeric_limits<float>::lowest();
		for (const auto & point : points)
		{
			x_min = std::min(x_min, point.x);
			y_min = std::min(y_min, point.y);
			x_max = std::max(x_max, point.x);
			y_max = std::max(y_max, point.y);
		}
		return cv::Rect2f(x_min, y_min, x_max - x_min, y_max - y_min);
	}


	static bool yolov9_box_candidate(const cv::Rect2f & before, const cv::Rect2f & after, const float scale, const float area_threshold)
	{
		constexpr float wh_threshold = 2.0f;
		constexpr float aspect_threshold = 100.0f;
		constexpr float eps = 1.0e-16f;
		const float w1 = before.width * scale;
		const float h1 = before.height * scale;
		const float w2 = after.width;
		const float h2 = after.height;
		if (w2 <= wh_threshold or h2 <= wh_threshold)
		{
			return false;
		}
		const float aspect = std::max(w2 / (h2 + eps), h2 / (w2 + eps));
		const float area_ratio = (w2 * h2) / (w1 * h1 + eps);
		return area_ratio > area_threshold and aspect < aspect_threshold;
	}


	static void yolov9_random_perspective(
		cv::Mat & image,
		std::vector<Yolov9Label> & labels,
		const float degrees,
		const float translate,
		const float scale,
		const float shear,
		const float perspective,
		const int border_y,
		const int border_x)
	{
		TAT(TATPARMS);

		const int height = image.rows + border_y * 2;
		const int width = image.cols + border_x * 2;
		if (height <= 0 or width <= 0)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid YOLOv9 augmentation output size %dx%d", width, height);
		}

		cv::Matx33d center = cv::Matx33d::eye();
		center(0, 2) = -image.cols / 2.0;
		center(1, 2) = -image.rows / 2.0;

		cv::Matx33d perspective_matrix = cv::Matx33d::eye();
		perspective_matrix(2, 0) = rand_uniform(-perspective, perspective);
		perspective_matrix(2, 1) = rand_uniform(-perspective, perspective);

		cv::Matx33d rotation = cv::Matx33d::eye();
		const double angle = rand_uniform(-degrees, degrees);
		const double rotation_scale = rand_uniform(1.0f - scale, 1.0f + scale);
		const cv::Mat rotation_2d = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), angle, rotation_scale);
		for (int row = 0; row < 2; ++row)
		{
			for (int col = 0; col < 3; ++col)
			{
				rotation(row, col) = rotation_2d.at<double>(row, col);
			}
		}

		cv::Matx33d shear_matrix = cv::Matx33d::eye();
		shear_matrix(0, 1) = std::tan(rand_uniform(-shear, shear) * CV_PI / 180.0f);
		shear_matrix(1, 0) = std::tan(rand_uniform(-shear, shear) * CV_PI / 180.0f);

		cv::Matx33d translation = cv::Matx33d::eye();
		translation(0, 2) = rand_uniform(0.5f - translate, 0.5f + translate) * width;
		translation(1, 2) = rand_uniform(0.5f - translate, 0.5f + translate) * height;

		const cv::Matx33d matrix = translation * shear_matrix * rotation * perspective_matrix * center;

		if (border_y != 0 or border_x != 0 or cv::norm(cv::Mat(matrix), cv::Mat(cv::Matx33d::eye())) > 1.0e-12)
		{
			cv::Mat transformed;
			if (perspective != 0.0f)
			{
				cv::warpPerspective(image, transformed, cv::Mat(matrix), cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			}
			else
			{
				cv::Mat affine(2, 3, CV_64F);
				for (int row = 0; row < 2; ++row)
				{
					for (int col = 0; col < 3; ++col)
					{
						affine.at<double>(row, col) = matrix(row, col);
					}
				}
				cv::warpAffine(image, transformed, affine, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			}
			image = transformed;
		}

		std::vector<Yolov9Label> kept;
		const bool use_segments = std::any_of(labels.begin(), labels.end(), [](const Yolov9Label & label)
		{
			return not label.segment.empty();
		});

		for (auto label : labels)
		{
			const cv::Rect2f before = label.box;
			if (use_segments and not label.segment.empty())
			{
				for (auto & point : label.segment)
				{
					point = transform_point(point, matrix, perspective != 0.0f);
					point.x = std::clamp(point.x, 0.0f, static_cast<float>(width));
					point.y = std::clamp(point.y, 0.0f, static_cast<float>(height));
				}
				label.box = rect_from_points(label.segment);
			}
			else
			{
				std::vector<cv::Point2f> points =
				{
					cv::Point2f(before.x, before.y),
					cv::Point2f(before.x + before.width, before.y + before.height),
					cv::Point2f(before.x, before.y + before.height),
					cv::Point2f(before.x + before.width, before.y)
				};
				for (auto & point : points)
				{
					point = transform_point(point, matrix, perspective != 0.0f);
				}
				label.box = rect_from_points(points);
			}

			clip_label(label, static_cast<float>(width), static_cast<float>(height));
			if (yolov9_box_candidate(before, label.box, static_cast<float>(rotation_scale), use_segments ? 0.01f : 0.10f))
			{
				kept.push_back(std::move(label));
			}
		}
		labels.swap(kept);
	}


	static float yolov9_ioa(const cv::Rect2f & box1, const cv::Rect2f & box2)
	{
		const float x1 = std::max(box1.x, box2.x);
		const float y1 = std::max(box1.y, box2.y);
		const float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
		const float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
		const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
		const float box2_area = std::max(0.0f, box2.width) * std::max(0.0f, box2.height) + 1.0e-7f;
		return intersection / box2_area;
	}


	static void yolov9_copy_paste(cv::Mat & image, std::vector<Yolov9Label> & labels, const float probability)
	{
		TAT(TATPARMS);

		if (probability <= 0.0f)
		{
			return;
		}

		std::vector<int> segmented;
		for (size_t idx = 0; idx < labels.size(); ++idx)
		{
			if (not labels[idx].segment.empty())
			{
				segmented.push_back(static_cast<int>(idx));
			}
		}

		if (segmented.empty())
		{
			return; // Matches the Python reference for box-only labels where len(segments) == 0.
		}

		std::vector<int> eligible;
		const float image_w = static_cast<float>(image.cols);
		for (const int idx : segmented)
		{
			const Yolov9Label & label = labels[idx];
			const cv::Rect2f mirrored(
				image_w - label.box.x - label.box.width,
				label.box.y,
				label.box.width,
				label.box.height);

			bool overlaps = false;
			for (const auto & other : labels)
			{
				if (yolov9_ioa(mirrored, other.box) >= 0.30f)
				{
					overlaps = true;
					break;
				}
			}
			if (not overlaps)
			{
				eligible.push_back(idx);
			}
		}

		if (eligible.empty())
		{
			return;
		}

		std::shuffle(eligible.begin(), eligible.end(), get_rnd_engine());
		const int copies = std::min<int>(eligible.size(), static_cast<int>(std::round(probability * eligible.size())));
		if (copies <= 0)
		{
			return;
		}

		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		const size_t original_label_count = labels.size();
		for (int i = 0; i < copies; ++i)
		{
			const Yolov9Label source = labels[eligible[i]];
			Yolov9Label pasted = source;
			pasted.box.x = image_w - source.box.x - source.box.width;
			for (auto & point : pasted.segment)
			{
				point.x = image_w - point.x;
			}
			labels.push_back(std::move(pasted));

			std::vector<cv::Point> contour;
			for (const auto & point : source.segment)
			{
				contour.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
			}
			if (contour.size() >= 3)
			{
				std::vector<std::vector<cv::Point>> contours = {contour};
				cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);
			}
		}

		if (labels.size() == original_label_count)
		{
			return;
		}

		cv::Mat flipped_image;
		cv::Mat flipped_mask;
		cv::flip(image, flipped_image, 1);
		cv::flip(mask, flipped_mask, 1);
		flipped_image.copyTo(image, flipped_mask);
	}


	static Yolov9Sample load_yolov9_mosaic_sample(const char * filename, const load_args & args)
	{
		TAT(TATPARMS);

		if (args.w != args.h)
		{
			darknet_fatal_error(DARKNET_LOC, "YOLOv9 mosaic augmentation requires square network dimensions, got %dx%d", args.w, args.h);
		}

		const int s = args.w;
		const int yc = static_cast<int>(rand_uniform(s * 0.5f, s * 1.5f));
		const int xc = static_cast<int>(rand_uniform(s * 0.5f, s * 1.5f));

		std::array<const char *, 4> selected =
		{
			filename,
			args.paths[rand_uint(0, args.m - 1)],
			args.paths[rand_uint(0, args.m - 1)],
			args.paths[rand_uint(0, args.m - 1)]
		};
		std::shuffle(selected.begin(), selected.end(), get_rnd_engine());

		Yolov9Sample mosaic;
		for (int i = 0; i < 4; ++i)
		{
			Yolov9Sample sample = load_yolov9_resized_sample(selected[i], args);
			const int h = sample.image.rows;
			const int w = sample.image.cols;
			if (i == 0)
			{
				mosaic.image = cv::Mat(s * 2, s * 2, sample.image.type(), cv::Scalar(114, 114, 114));
			}

			int x1a = 0;
			int y1a = 0;
			int x2a = 0;
			int y2a = 0;
			int x1b = 0;
			int y1b = 0;
			int x2b = 0;
			int y2b = 0;

			if (i == 0)
			{
				x1a = std::max(xc - w, 0);
				y1a = std::max(yc - h, 0);
				x2a = xc;
				y2a = yc;
				x1b = w - (x2a - x1a);
				y1b = h - (y2a - y1a);
				x2b = w;
				y2b = h;
			}
			else if (i == 1)
			{
				x1a = xc;
				y1a = std::max(yc - h, 0);
				x2a = std::min(xc + w, s * 2);
				y2a = yc;
				x1b = 0;
				y1b = h - (y2a - y1a);
				x2b = std::min(w, x2a - x1a);
				y2b = h;
			}
			else if (i == 2)
			{
				x1a = std::max(xc - w, 0);
				y1a = yc;
				x2a = xc;
				y2a = std::min(s * 2, yc + h);
				x1b = w - (x2a - x1a);
				y1b = 0;
				x2b = w;
				y2b = std::min(y2a - y1a, h);
			}
			else
			{
				x1a = xc;
				y1a = yc;
				x2a = std::min(xc + w, s * 2);
				y2a = std::min(s * 2, yc + h);
				x1b = 0;
				y1b = 0;
				x2b = std::min(w, x2a - x1a);
				y2b = std::min(y2a - y1a, h);
			}

			sample.image(cv::Rect(x1b, y1b, x2b - x1b, y2b - y1b)).copyTo(mosaic.image(cv::Rect(x1a, y1a, x2a - x1a, y2a - y1a)));
			const float padw = static_cast<float>(x1a - x1b);
			const float padh = static_cast<float>(y1a - y1b);
			for (auto & label : sample.labels)
			{
				label.box.x += padw;
				label.box.y += padh;
				for (auto & point : label.segment)
				{
					point.x += padw;
					point.y += padh;
				}
				clip_label(label, static_cast<float>(s * 2), static_cast<float>(s * 2));
				if (label_area(label) > 0.0f)
				{
					mosaic.labels.push_back(std::move(label));
				}
			}
		}

		yolov9_copy_paste(mosaic.image, mosaic.labels, args.copy_paste_prob);
		yolov9_random_perspective(mosaic.image, mosaic.labels, args.degrees, args.translate, args.yolov9_scale, args.shear, args.perspective, -s / 2, -s / 2);
		return mosaic;
	}


	static Yolov9Sample load_yolov9_letterbox_sample(const char * filename, const load_args & args)
	{
		TAT(TATPARMS);

		Yolov9Sample sample = load_yolov9_resized_sample(filename, args);
		Yolov9Letterbox letterbox = yolov9_letterbox_rgb(sample.image, args.w, args.h);
		sample.image = letterbox.image;
		apply_letterbox_to_labels(sample.labels, letterbox);
		yolov9_random_perspective(sample.image, sample.labels, args.degrees, args.translate, args.yolov9_scale, args.shear, args.perspective, 0, 0);
		return sample;
	}


	static void yolov9_mixup(Yolov9Sample & sample, const Yolov9Sample & other)
	{
		TAT(TATPARMS);

		std::gamma_distribution<float> gamma(32.0f, 1.0f);
		float a = gamma(get_rnd_engine());
		float b = gamma(get_rnd_engine());
		if (a <= 0.0f and b <= 0.0f)
		{
			a = 1.0f;
			b = 1.0f;
		}
		const float ratio = a / (a + b);

		for (int y = 0; y < sample.image.rows; ++y)
		{
			cv::Vec3b * dst = sample.image.ptr<cv::Vec3b>(y);
			const cv::Vec3b * src = other.image.ptr<cv::Vec3b>(y);
			for (int x = 0; x < sample.image.cols; ++x)
			{
				for (int channel = 0; channel < 3; ++channel)
				{
					dst[x][channel] = static_cast<unsigned char>(dst[x][channel] * ratio + src[x][channel] * (1.0f - ratio));
				}
			}
		}

		sample.labels.insert(sample.labels.end(), other.labels.begin(), other.labels.end());
	}


	static std::vector<std::array<float, 5>> normalize_yolov9_labels(std::vector<Yolov9Label> & labels, const int width, const int height)
	{
		TAT(TATPARMS);

		std::vector<std::array<float, 5>> normalized;
		const float clip_w = std::max(0.0f, width - 1.0e-3f);
		const float clip_h = std::max(0.0f, height - 1.0e-3f);
		for (auto & label : labels)
		{
			clip_label(label, clip_w, clip_h);
			if (label.box.width <= 0.0f or label.box.height <= 0.0f)
			{
				continue;
			}
			normalized.push_back({
				static_cast<float>(label.cls),
				(label.box.x + label.box.width / 2.0f) / width,
				(label.box.y + label.box.height / 2.0f) / height,
				label.box.width / width,
				label.box.height / height
			});
		}
		return normalized;
	}


	static void yolov9_augment_hsv_rgb(cv::Mat & image, const float hgain, const float sgain, const float vgain)
	{
		TAT(TATPARMS);

		if ((hgain == 0.0f and sgain == 0.0f and vgain == 0.0f) or image.channels() < 3)
		{
			return;
		}

		const float hue_gain = rand_uniform(-1.0f, 1.0f) * hgain + 1.0f;
		const float sat_gain = rand_uniform(-1.0f, 1.0f) * sgain + 1.0f;
		const float val_gain = rand_uniform(-1.0f, 1.0f) * vgain + 1.0f;

		cv::Mat hsv;
		cv::cvtColor(image, hsv, cv::COLOR_RGB2HSV);
		std::vector<cv::Mat> channels;
		cv::split(hsv, channels);

		cv::Mat lut_hue(1, 256, CV_8UC1);
		cv::Mat lut_sat(1, 256, CV_8UC1);
		cv::Mat lut_val(1, 256, CV_8UC1);
		for (int i = 0; i < 256; ++i)
		{
			lut_hue.at<unsigned char>(i) = static_cast<unsigned char>(static_cast<int>(i * hue_gain) % 180);
			lut_sat.at<unsigned char>(i) = static_cast<unsigned char>(std::clamp(i * sat_gain, 0.0f, 255.0f));
			lut_val.at<unsigned char>(i) = static_cast<unsigned char>(std::clamp(i * val_gain, 0.0f, 255.0f));
		}

		cv::LUT(channels[0], lut_hue, channels[0]);
		cv::LUT(channels[1], lut_sat, channels[1]);
		cv::LUT(channels[2], lut_val, channels[2]);
		cv::merge(channels, hsv);
		cv::cvtColor(hsv, image, cv::COLOR_HSV2RGB);
	}


	static void apply_yolov9_flips(cv::Mat & image, std::vector<std::array<float, 5>> & labels, const float flipud_prob, const float fliplr_prob)
	{
		TAT(TATPARMS);

		if (flipud_prob > 0.0f and rand_uniform(0.0f, 1.0f) < flipud_prob)
		{
			cv::flip(image, image, 0);
			for (auto & label : labels)
			{
				label[2] = 1.0f - label[2];
			}
		}

		if (fliplr_prob > 0.0f and rand_uniform(0.0f, 1.0f) < fliplr_prob)
		{
			cv::flip(image, image, 1);
			for (auto & label : labels)
			{
				label[1] = 1.0f - label[1];
			}
		}
	}


	static void write_yolov9_truth(const std::vector<std::array<float, 5>> & labels, const int max_boxes, const int truth_size, float * truth)
	{
		TAT(TATPARMS);

		int count = 0;
		for (const auto & label : labels)
		{
			if (count >= max_boxes)
			{
				break;
			}
			if (label[3] <= 0.0f or label[4] <= 0.0f)
			{
				continue;
			}
			float * truth_ptr = truth + count * truth_size;
			truth_ptr[0] = label[1];
			truth_ptr[1] = label[2];
			truth_ptr[2] = label[3];
			truth_ptr[3] = label[4];
			truth_ptr[4] = label[0];
			if (truth_size > 5)
			{
				truth_ptr[5] = count + 1;
			}
			++count;
		}
	}
}


list *get_paths(const char *filename)
{
	TAT(TATPARMS);

	char *path;
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		file_error(filename, DARKNET_LOC);
	}

	list *lines = make_list();
	while((path=fgetl(file)))
	{
		list_insert(lines, path);
	}
	fclose(file);

	if (lines->size == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to read any lines from %s", filename);
	}

	return lines;
}

char **get_sequential_paths(char **paths, int n, int m, int mini_batch, int augment_speed, int contrastive)
{
	TAT(TATPARMS);

	int speed = rand_int(1, augment_speed);
	if (speed < 1)
	{
		speed = 1;
	}

	char** sequentia_paths = (char**)xcalloc(n, sizeof(char*));

	unsigned int *start_time_indexes = (unsigned int *)xcalloc(mini_batch, sizeof(unsigned int));
	for (int i = 0; i < mini_batch; ++i)
	{
		if (contrastive && (i % 2) == 1)
		{
			start_time_indexes[i] = start_time_indexes[i - 1];
		}
		else
		{
			start_time_indexes[i] = rand_uint(0, m - 1);
		}
	}

	for (int i = 0; i < n; ++i)
	{
		int time_line_index = i % mini_batch;
		unsigned int index = start_time_indexes[time_line_index] % m;
		start_time_indexes[time_line_index] += speed;

		sequentia_paths[i] = paths[index];
	}
	free(start_time_indexes);

	return sequentia_paths;
}


char **get_random_paths_custom(char **paths, int n, int m, int contrastive)
{
	TAT(TATPARMS);

	char** random_paths = (char**)xcalloc(n, sizeof(char*));

	int old_index = 0;

	// "n" is the total number of filenames to be returned at once
	for (int i = 0; i < n; ++i)
	{
		int index = rand_uint(0, m - 1);
		if (contrastive && (i % 2 == 1))
		{
			index = old_index;
		}
		else
		{
			old_index = index;
		}
		random_paths[i] = paths[index];
	}

	return random_paths;
}


char **get_random_paths(char **paths, int n, int m)
{
	TAT(TATPARMS);

	return get_random_paths_custom(paths, n, m, 0);
}


box_label *read_boxes(const char *filename, int *n)
{
	TAT(TATPARMS);

	if (filename == nullptr or filename[0] == '\0')
	{
		darknet_fatal_error(DARKNET_LOC, "failed to open annotation file \"\" (no filename given)");
	}

	if (n == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "invalid \"n\" pointer while processing annotation file \"%s\"", filename);
	}
	*n = 0;

	std::FILE *file = std::fopen(filename, "r");
	if (not file)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to open annotation file \"%s\"", filename);
	}

	const int max_obj_img	= 4000;// 30000;
	const int img_hash		= (custom_hash(filename) % max_obj_img) * max_obj_img;
	float x					= 0.0f;
	float y					= 0.0f;
	float w					= 0.0f;
	float h					= 0.0f;
	int id					= 0;
	int line_counter		= 0;
	box_label * boxes		= nullptr;

	while (std::fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
	{
//		*cfg_and_state.output << "x=" << x << " y=" << y << " w=" << w << " h=" << h << std::endl;

		boxes = reinterpret_cast<box_label*>(xrealloc(boxes, (line_counter + 1) * sizeof(box_label)));
		box_label & box	= boxes[line_counter];
		box.track_id	= line_counter + img_hash;
		box.id			= id;
		box.x			= std::clamp(x				, 0.0f, 1.0f);
		box.y			= std::clamp(y				, 0.0f, 1.0f);
		box.h			= std::clamp(h				, 0.0f, 1.0f);
		box.w			= std::clamp(w				, 0.0f, 1.0f);
		box.left		= std::clamp(x - w / 2.0f	, 0.0f, 1.0f);
		box.right		= std::clamp(x + w / 2.0f	, 0.0f, 1.0f);
		box.top			= std::clamp(y - h / 2.0f	, 0.0f, 1.0f);
		box.bottom		= std::clamp(y + h / 2.0f	, 0.0f, 1.0f);

		if (id < 0) // no simple way to get the maximum number of classes at this point in the code
		{
			darknet_fatal_error(DARKNET_LOC, "invalid class id in \"%s\" (line #%d, id=%d)", filename, line_counter + 1, id);
		}

		if (box.left	<	0.0f or
			box.right	>	1.0f or
			box.top		<	0.0f or
			box.bottom	>	1.0f or
			box.w		<=	0.0f or
			box.h		<=	0.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid coordinate in \"%s\" (line #%d, cx=%f, cy=%f, width=%f, height=%f, left=%f, right=%f, top=%f, bottom=%f)",
					filename,
					line_counter + 1,
					box.x,
					box.y,
					box.h,
					box.w,
					box.left,
					box.right,
					box.top,
					box.bottom);
		}

		++line_counter;
	}

	std::fclose(file);
	*n = line_counter;

	return boxes;
}


void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < n; ++i)
	{
		if(boxes[i].x == 0 && boxes[i].y == 0)
		{
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
			(boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
		{
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		boxes[i].left   = boxes[i].left  * sx - dx;
		boxes[i].right  = boxes[i].right * sx - dx;
		boxes[i].top    = boxes[i].top   * sy - dy;
		boxes[i].bottom = boxes[i].bottom* sy - dy;

		if(flip)
		{
			float swap = boxes[i].left;
			boxes[i].left = 1.0f - boxes[i].right;
			boxes[i].right = 1.0f - swap;
		}

		boxes[i].left =  constrain(0, 1, boxes[i].left);
		boxes[i].right = constrain(0, 1, boxes[i].right);
		boxes[i].top =   constrain(0, 1, boxes[i].top);
		boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

		boxes[i].x = (boxes[i].left+boxes[i].right)/2;
		boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain(0, 1, boxes[i].w);
		boxes[i].h = constrain(0, 1, boxes[i].h);
	}
}


int fill_truth_detection(const char *path, int num_boxes, int truth_size, float *truth, int classes, int flip, float dx, float dy, float sx, float sy, int net_w, int net_h)
{
	TAT(TATPARMS);

	// This method is used during the training process to load the boxes for the given image.

	char labelpath[4096];
	replace_image_to_label(path, labelpath);

	int count = 0;
	int i;
	box_label *boxes = read_boxes(labelpath, &count);
	int min_w_h = 0;
	float lowest_w = 1.F / net_w;
	float lowest_h = 1.F / net_h;

	std::shuffle(boxes, boxes + count, get_rnd_engine());

	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	if (count > num_boxes)
	{
		count = num_boxes;
	}
	float x, y, w, h;
	int id;
	int sub = 0;

	for (i = 0; i < count; ++i)
	{
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;
		int track_id = boxes[i].track_id;

		// not detect small objects
		//if ((w < 0.001F || h < 0.001F)) continue;
		// if truth (box for object) is smaller than 1x1 pix
		//char buff[256];
		if (id >= classes)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d in %s", id, labelpath);
		}
		if ((w < lowest_w || h < lowest_h))
		{
			++sub;
			continue;
		}

		if (x == 999999 || y == 999999)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid annotation for class ID #%d in %s", id, labelpath);
		}
		/// @todo shouldn't this be x - w/2 < 0.0f?  And same for other variables?
		if (x <= 0.0f || x > 1.0f || y <= 0.0f || y > 1.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid coordinates for class ID #%d in %s", id, labelpath);
		}
		/// @todo again, instead of checking for > 1, shouldn't we check x + w / 2 ?
		if (w > 1.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid width for class ID #%d in %s", id, labelpath);
		}
		/// @todo check for y - h/2 and y + h/2?
		if (h > 1.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid height for class ID #%d in %s", id, labelpath);
		}

		if (x == 0) x += lowest_w;
		if (y == 0) y += lowest_h;

		truth[(i-sub)*truth_size +0] = x;
		truth[(i-sub)*truth_size +1] = y;
		truth[(i-sub)*truth_size +2] = w;
		truth[(i-sub)*truth_size +3] = h;
		truth[(i-sub)*truth_size +4] = id;
		truth[(i-sub)*truth_size +5] = track_id;

		if (min_w_h == 0) min_w_h = w*net_w;
		if (min_w_h > w*net_w) min_w_h = w*net_w;
		if (min_w_h > h*net_h) min_w_h = h*net_h;
	}
	free(boxes);
	return min_w_h;
}


void Darknet::free_data(data & d)
{
	TAT_REVIEWED(TATPARMS, "2024-04-04");

	// this is the only place in the entire codebase where the "shallow" flag is checked
	if (d.shallow == 0)
	{
		free_matrix(d.X);
		free_matrix(d.y);
	}
	else
	{
		free(d.X.vals);
		free(d.y.vals);
		d.X.vals = nullptr;
		d.y.vals = nullptr;
	}
	return;
}


void blend_truth(float *new_truth, int boxes, int truth_size, float *old_truth)
{
	TAT(TATPARMS);

	int count_new_truth = 0;
	int t;
	for (t = 0; t < boxes; ++t)
	{
		float x = new_truth[t*truth_size];
		if (!x)
		{
			break;
		}
		count_new_truth++;

	}
	for (t = count_new_truth; t < boxes; ++t)
	{
		float *new_truth_ptr = new_truth + t*truth_size;
		float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
		float x = old_truth_ptr[0];
		if (!x)
		{
			break;
		}

		new_truth_ptr[0] = old_truth_ptr[0];
		new_truth_ptr[1] = old_truth_ptr[1];
		new_truth_ptr[2] = old_truth_ptr[2];
		new_truth_ptr[3] = old_truth_ptr[3];
		new_truth_ptr[4] = old_truth_ptr[4];
	}
}


void blend_truth_mosaic(float *new_truth, int boxes, int truth_size, float *old_truth, int w, int h, float cut_x, float cut_y, int i_mixup, int left_shift, int right_shift, int top_shift, int bot_shift, int net_w, int net_h, int mosaic_bound)
{
	TAT(TATPARMS);

	const float lowest_w = 1.F / net_w;
	const float lowest_h = 1.F / net_h;

	int count_new_truth = 0;
	int t;
	for (t = 0; t < boxes; ++t)
	{
		float x = new_truth[t*truth_size];
		if (!x)
		{
			break;
		}
		count_new_truth++;

	}
	int new_t = count_new_truth;
	for (t = count_new_truth; t < boxes; ++t)
	{
		float *new_truth_ptr = new_truth + new_t*truth_size;
		new_truth_ptr[0] = 0;
		float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
		float x = old_truth_ptr[0];
		if (!x)
		{
			break;
		}

		float xb = old_truth_ptr[0];
		float yb = old_truth_ptr[1];
		float wb = old_truth_ptr[2];
		float hb = old_truth_ptr[3];



		// shift 4 images
		if (i_mixup == 0)
		{
			xb = xb - (float)(w - cut_x - right_shift) / w;
			yb = yb - (float)(h - cut_y - bot_shift) / h;
		}
		if (i_mixup == 1)
		{
			xb = xb + (float)(cut_x - left_shift) / w;
			yb = yb - (float)(h - cut_y - bot_shift) / h;
		}
		if (i_mixup == 2)
		{
			xb = xb - (float)(w - cut_x - right_shift) / w;
			yb = yb + (float)(cut_y - top_shift) / h;
		}
		if (i_mixup == 3)
		{
			xb = xb + (float)(cut_x - left_shift) / w;
			yb = yb + (float)(cut_y - top_shift) / h;
		}

		int left = (xb - wb / 2)*w;
		int right = (xb + wb / 2)*w;
		int top = (yb - hb / 2)*h;
		int bot = (yb + hb / 2)*h;

		if(mosaic_bound)
		{
			// fix out of Mosaic-bound
			float left_bound = 0, right_bound = 0, top_bound = 0, bot_bound = 0;
			if (i_mixup == 0)
			{
				left_bound = 0;
				right_bound = cut_x;
				top_bound = 0;
				bot_bound = cut_y;
			}
			if (i_mixup == 1)
			{
				left_bound = cut_x;
				right_bound = w;
				top_bound = 0;
				bot_bound = cut_y;
			}
			if (i_mixup == 2)
			{
				left_bound = 0;
				right_bound = cut_x;
				top_bound = cut_y;
				bot_bound = h;
			}
			if (i_mixup == 3)
			{
				left_bound = cut_x;
				right_bound = w;
				top_bound = cut_y;
				bot_bound = h;
			}


			if (left < left_bound)
			{
				left = left_bound;
			}
			if (right > right_bound)
			{
				right = right_bound;
			}
			if (top < top_bound) top = top_bound;
			if (bot > bot_bound) bot = bot_bound;


			xb = ((float)(right + left) / 2) / w;
			wb = ((float)(right - left)) / w;
			yb = ((float)(bot + top) / 2) / h;
			hb = ((float)(bot - top)) / h;
		}
		else
		{
			// fix out of bound
			if (left < 0)
			{
				float diff = (float)left / w;
				xb = xb - diff / 2;
				wb = wb + diff;
			}

			if (right > w)
			{
				float diff = (float)(right - w) / w;
				xb = xb - diff / 2;
				wb = wb - diff;
			}

			if (top < 0)
			{
				float diff = (float)top / h;
				yb = yb - diff / 2;
				hb = hb + diff;
			}

			if (bot > h)
			{
				float diff = (float)(bot - h) / h;
				yb = yb - diff / 2;
				hb = hb - diff;
			}

			left = (xb - wb / 2)*w;
			right = (xb + wb / 2)*w;
			top = (yb - hb / 2)*h;
			bot = (yb + hb / 2)*h;
		}


		// leave only within the image
		if(left >= 0 && right <= w && top >= 0 && bot <= h &&
			wb > 0 && wb < 1 && hb > 0 && hb < 1 &&
			xb > 0 && xb < 1 && yb > 0 && yb < 1 &&
			wb > lowest_w && hb > lowest_h)
		{
			new_truth_ptr[0] = xb;
			new_truth_ptr[1] = yb;
			new_truth_ptr[2] = wb;
			new_truth_ptr[3] = hb;
			new_truth_ptr[4] = old_truth_ptr[4];
			new_t++;
		}
	}
}


data load_data_detection_yolov9(load_args args)
{
	TAT(TATPARMS);

	args.c = args.c ? args.c : 3;
	if (args.c != 3)
	{
		darknet_fatal_error(DARKNET_LOC, "YOLOv9 augmentation currently requires 3-channel RGB images, got %d channels", args.c);
	}

	data d = {0};
	d.shallow = 0;
	d.X.rows = args.n;
	d.X.cols = args.h * args.w * args.c;
	d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
	d.y = make_matrix(args.n, args.truth_size * args.num_boxes);

	char **random_paths = nullptr;
	if (args.track)
	{
		random_paths = get_sequential_paths(args.paths, args.n, args.m, args.mini_batch, args.augment_speed, args.contrastive);
	}
	else
	{
		random_paths = get_random_paths_custom(args.paths, args.n, args.m, args.contrastive);
	}

	for (int i = 0; i < args.n; ++i)
	{
		const char *filename = random_paths[i];
		const bool use_mosaic = args.mosaic_prob > 0.0f and rand_uniform(0.0f, 1.0f) < args.mosaic_prob;
		Yolov9Sample sample = use_mosaic
			? load_yolov9_mosaic_sample(filename, args)
			: load_yolov9_letterbox_sample(filename, args);

		if (use_mosaic and args.mixup_prob > 0.0f and rand_uniform(0.0f, 1.0f) < args.mixup_prob)
		{
			const char * mixup_filename = args.paths[rand_uint(0, args.m - 1)];
			Yolov9Sample other = load_yolov9_mosaic_sample(mixup_filename, args);
			yolov9_mixup(sample, other);
		}

		std::vector<std::array<float, 5>> labels = normalize_yolov9_labels(sample.labels, sample.image.cols, sample.image.rows);
		yolov9_augment_hsv_rgb(sample.image, args.hsv_h, args.hsv_s, args.hsv_v);
		apply_yolov9_flips(sample.image, labels, args.flipud_prob, args.fliplr_prob);
		write_yolov9_truth(labels, args.num_boxes, args.truth_size, d.y.vals[i]);

		Darknet::Image ai = Darknet::rgb_mat_to_rgb_image(sample.image);
		d.X.vals[i] = ai.data;

		if (args.show_imgs)
		{
			const int random_index = rand_uint();
			Darknet::Image tmp_ai = Darknet::copy_image(ai);
			char buff[1000];
			sprintf(buff, "aug_yolov9_%d_%d_%d", random_index, i, rand_uint());
			for (int t = 0; t < args.num_boxes; ++t)
			{
				Darknet::Box b = float_to_box_stride(d.y.vals[i] + t * args.truth_size, 1);
				if (!b.x) break;
				int left = (b.x - b.w / 2.0f) * ai.w;
				int right = (b.x + b.w / 2.0f) * ai.w;
				int top = (b.y - b.h / 2.0f) * ai.h;
				int bot = (b.y + b.h / 2.0f) * ai.h;
				Darknet::draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50);
			}
			Darknet::save_image(tmp_ai, buff);
			if (args.show_imgs == 1)
			{
				Darknet::show_image(tmp_ai, buff);
				cv::waitKey(0);
			}
			Darknet::free_image(tmp_ai);
		}
	}

	if (random_paths)
	{
		free(random_paths);
	}

	return d;
}


data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int use_gaussian_noise, int use_blur, int use_mixup,
	float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs)
{
	TAT(TATPARMS);

	// This is the method that gets called to load the "n" images for each loading thread while training a network.

	c = c ? c : 3;

	if (use_mixup == 2 || use_mixup == 4)
	{
		darknet_fatal_error(DARKNET_LOC, "cutmix=1 isn't supported for detector");
	}

	if (use_mixup == 3 && letter_box)
	{
		darknet_fatal_error(DARKNET_LOC, "letterbox and mosaic cannot be combined");
	}

	if (rand_bool())
	{
		use_mixup = 0;
	}

	int *cut_x = nullptr;
	int *cut_y = nullptr;

	if (use_mixup == 3)
	{
		cut_x = (int*)calloc(n, sizeof(int));
		cut_y = (int*)calloc(n, sizeof(int));
		const float min_offset = 0.2; // 20%

		for (int i = 0; i < n; ++i)
		{
			cut_x[i] = rand_int(w*min_offset, w*(1 - min_offset));
			cut_y[i] = rand_int(h*min_offset, h*(1 - min_offset));
		}
	}

	data d = {0};
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
	d.X.cols = h*w*c;

	float r1 = 0.0f;
	float r2 = 0.0f;
	float r3 = 0.0f;
	float r4 = 0.0f;
	float resize_r1 = 0.0f;
	float resize_r2 = 0.0f;
	float dhue = 0.0f;
	float dsat = 0.0f;
	float dexp = 0.0f;
	float flip = 0.0f;
	float blur = 0.0f;
	int augmentation_calculated = 0;
	int gaussian_noise = 0;

	d.y = make_matrix(n, truth_size * boxes);

	for (int i_mixup = 0; i_mixup <= use_mixup; i_mixup++)
	{
		if (i_mixup)
		{
			augmentation_calculated = 0;   // recalculate augmentation for the 2nd sequence if(track==1)
		}

		char **random_paths;
		if (track)
		{
			random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
		}
		else
		{
			random_paths = get_random_paths_custom(paths, n, m, contrastive);
		}

		// about to load multiple images ("n"), usually batch size divided by the number of loading threads
		for (int i = 0; i < n; ++i)
		{
			float *truth = (float*)xcalloc(truth_size * boxes, sizeof(float));
			const char *filename = random_paths[i];

			cv::Mat src = load_rgb_mat_image(filename, c);

			const int oh = src.rows;	// original height
			const int ow = src.cols;	// original width

			int dw = (ow*jitter);
			int dh = (oh*jitter);

			float resize_down = resize;
			float resize_up = resize;

			if (resize_down > 1.0f)
			{
				resize_down = 1.0f / resize_down;
			}
			const int min_rdw = ow *(1.0f - (1.0f / resize_down)) / 2.0f;   // < 0
			const int min_rdh = oh *(1.0f - (1.0f / resize_down)) / 2.0f;   // < 0

			if (resize_up < 1.0f)
			{
				resize_up = 1.0f / resize_up;
			}
			const int max_rdw = ow * (1.0f - (1.0f / resize_up)) / 2.0f;     // > 0
			const int max_rdh = oh * (1.0f - (1.0f / resize_up)) / 2.0f;     // > 0

			if (!augmentation_calculated || !track)
			{
				augmentation_calculated = 1;
				resize_r1 = rand_float();
				resize_r2 = rand_float();

				if (!contrastive || contrastive_jit_flip || i % 2 == 0)
				{
					r1 = rand_float();
					r2 = rand_float();
					r3 = rand_float();
					r4 = rand_float();

					flip = use_flip ? rand_bool() : 0;
				}

				if (!contrastive || contrastive_color || i % 2 == 0)
				{
					dhue = rand_uniform(-hue, hue);
					dsat = rand_scale(saturation);
					dexp = rand_scale(exposure);
				}

				if (use_blur)
				{
					int tmp_blur = rand_int(0, 2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
					if (tmp_blur == 0)
					{
						blur = 0;
					}
					else if (tmp_blur == 1)
					{
						blur = 1;
					}
					else
					{
						blur = use_blur;
					}
				}

				if (use_gaussian_noise && rand_bool())
				{
					gaussian_noise = use_gaussian_noise;
				}
				else
				{
					gaussian_noise = 0;
				}
			}

			int pleft = rand_precalc_random(-dw, dw, r1);
			int pright = rand_precalc_random(-dw, dw, r2);
			int ptop = rand_precalc_random(-dh, dh, r3);
			int pbot = rand_precalc_random(-dh, dh, r4);

			if (resize < 1.0f)
			{
				// downsize only
				pleft += rand_precalc_random(min_rdw, 0, resize_r1);
				pright += rand_precalc_random(min_rdw, 0, resize_r2);
				ptop += rand_precalc_random(min_rdh, 0, resize_r1);
				pbot += rand_precalc_random(min_rdh, 0, resize_r2);
			}
			else
			{
				pleft += rand_precalc_random(min_rdw, max_rdw, resize_r1);
				pright += rand_precalc_random(min_rdw, max_rdw, resize_r2);
				ptop += rand_precalc_random(min_rdh, max_rdh, resize_r1);
				pbot += rand_precalc_random(min_rdh, max_rdh, resize_r2);
			}

			if (letter_box)
			{
				float img_ar = (float)ow / (float)oh;
				float net_ar = (float)w / (float)h;
				float result_ar = img_ar / net_ar;
				if (result_ar > 1)  // sheight - should be increased
				{
					float oh_tmp = ow / net_ar;
					float delta_h = (oh_tmp - oh)/2;
					ptop = ptop - delta_h;
					pbot = pbot - delta_h;
				}
				else  // swidth - should be increased
				{
					float ow_tmp = oh * net_ar;
					float delta_w = (ow_tmp - ow)/2;
					pleft = pleft - delta_w;
					pright = pright - delta_w;
				}
			}

			// move each 2nd image to the corner - so that most of it was visible
			if (use_mixup == 3 && rand_bool())
			{
				if (flip)
				{
					if (i_mixup == 0) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 1) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 2) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
					if (i_mixup == 3) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
				}
				else
				{
					if (i_mixup == 0) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 1) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
					if (i_mixup == 2) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
					if (i_mixup == 3) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
				}
			}

			const int swidth = ow - pleft - pright;
			const int sheight = oh - ptop - pbot;

			const float sx = (float)swidth / ow;
			const float sy = (float)sheight / oh;

			const float dx = ((float)pleft / ow) / sx;
			const float dy = ((float)ptop / oh) / sy;

			// This is where we get the annotations for this image.
			const int min_w_h = fill_truth_detection(filename, boxes, truth_size, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

			if ((min_w_h / 8) < blur && blur > 1)
			{
				blur = min_w_h / 8;   // disable blur if one of the objects is too small
			}

			Darknet::Image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur, boxes, truth_size, truth);

			if (use_mixup == 0)
			{
				d.X.vals[i] = ai.data;
				memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
			}
			else if (use_mixup == 1)
			{
				if (i_mixup == 0)
				{
					d.X.vals[i] = ai.data;
					memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
				}
				else if (i_mixup == 1)
				{
					Darknet::Image old_img = make_empty_image(w, h, c);
					old_img.data = d.X.vals[i];
					blend_images_cv(ai, 0.5, old_img, 0.5);
					blend_truth(d.y.vals[i], boxes, truth_size, truth);
					Darknet::free_image(old_img);
					d.X.vals[i] = ai.data;
				}
			}
			else if (use_mixup == 3)
			{
				if (i_mixup == 0)
				{
					Darknet::Image tmp_img = make_image(w, h, c);
					d.X.vals[i] = tmp_img.data;
				}

				if (flip)
				{
					int tmp = pleft;
					pleft = pright;
					pright = tmp;
				}

				const int left_shift	= std::min(cut_x[i]		, std::max(0, (-pleft * w	/ ow)));
				const int top_shift		= std::min(cut_y[i]		, std::max(0, (-ptop * h	/ oh)));
				const int right_shift	= std::min(w - cut_x[i]	, std::max(0, (-pright * w	/ ow)));
				const int bot_shift		= std::min(h - cut_y[i]	, std::max(0, (-pbot * h	/ oh)));

				//int k, x, y;
				for (int k = 0; k < c; ++k)
				{
					for (int y = 0; y < h; ++y)
					{
						int j = y*w + k*w*h;
						if (i_mixup == 0 && y < cut_y[i])
						{
							int j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
							memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
						}
						if (i_mixup == 1 && y < cut_y[i])
						{
							int j_src = left_shift + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
							memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w-cut_x[i]) * sizeof(float));
						}
						if (i_mixup == 2 && y >= cut_y[i])
						{
							int j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i])*w + k*w*h;
							memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
						}
						if (i_mixup == 3 && y >= cut_y[i])
						{
							int j_src = left_shift + (top_shift + y - cut_y[i])*w + k*w*h;
							memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
						}
					}
				}

				blend_truth_mosaic(d.y.vals[i], boxes, truth_size, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift, right_shift, top_shift, bot_shift, w, h, mosaic_bound);

				Darknet::free_image(ai);
				ai.data = d.X.vals[i];
			}

			if (show_imgs && i_mixup == use_mixup)   // delete i_mixup
			{
				const int random_index = rand_uint();

				Darknet::Image tmp_ai = Darknet::copy_image(ai);
				char buff[1000];
				sprintf(buff, "aug_%d_%d_%d", random_index, i, rand_uint());
				int t;
				for (t = 0; t < boxes; ++t)
				{
					Darknet::Box b = float_to_box_stride(d.y.vals[i] + t*truth_size, 1);
					if (!b.x) break;
					int left = (b.x - b.w / 2.)*ai.w;
					int right = (b.x + b.w / 2.)*ai.w;
					int top = (b.y - b.h / 2.)*ai.h;
					int bot = (b.y + b.h / 2.)*ai.h;
					Darknet::draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
				}

				Darknet::save_image(tmp_ai, buff);
				if (show_imgs == 1)
				{
					Darknet::show_image(tmp_ai, buff);
					cv::waitKey(0);
				}
				Darknet::free_image(tmp_ai);
			}

			free(truth);
		}

		if (random_paths)
		{
			free(random_paths);
		}
	}

	return d;
}


void Darknet::load_single_image_data(load_args args)
{
	TAT(TATPARMS);

	// Note:  even though the name is load_single_image_data() note that this will likely result in more than 1 image
	// loaded due to the args.n parameter.

	if (args.aspect		== 0.0f)	args.aspect		= 1.0f;
	if (args.exposure	== 0.0f)	args.exposure	= 1.0f;
	if (args.saturation	== 0.0f)	args.saturation	= 1.0f;

	switch (args.type)
	{
		case IMAGE_DATA:
		{
			// 2024:  used in coco.cpp, detector.cpp, yolo.cpp
			/// @todo 2025-11-19: Why not call Darknet::load_image() with the desired width/height?  Darknet::resize_image() is slower than OpenCV.
			*(args.im) = Darknet::load_image(args.path, 0, 0, args.c);
			*(args.resized) = Darknet::resize_image(*(args.im), args.w, args.h);
			break;
		}
		case LETTERBOX_DATA:
		{
			// 2024:  used in detector.cpp
			*(args.im) = Darknet::load_image(args.path, 0, 0, args.c);
			*(args.resized) = Darknet::letterbox_image(*(args.im), args.w, args.h);
			break;
		}
		case DETECTION_DATA:
		{
			// 2024:  used in detector.cpp (when training a neural network)
			if (args.augment_policy == 1)
			{
				*args.d = load_data_detection_yolov9(args);
			}
			else
			{
				*args.d = load_data_detection(args.n, args.paths, args.m, args.w, args.h, args.c, args.num_boxes, args.truth_size, args.classes, args.flip, args.gaussian_noise, args.blur, args.mixup, args.jitter, args.resize,
						args.hue, args.saturation, args.exposure, args.mini_batch, args.track, args.augment_speed, args.letter_box, args.mosaic_bound, args.contrastive, args.contrastive_jit_flip, args.contrastive_color, args.show_imgs);
			}
			break;
		}
	}

	return;
}


void Darknet::image_loading_loop(const int idx, load_args args)
{
	TAT_REVIEWED(TATPARMS, "2024-04-11");

	/* This loop runs on a secondary thread.  There are several of these threads started when the training starts,
	 * and they stay active throughout the training process, until stop_image_loading_threads() is eventually called.
	 */

	const std::string name = "image loading loop #" + std::to_string(idx);
	cfg_and_state.set_thread_name(name);

	const int number_of_threads	= args.threads;	// typically will be 6
	const int number_of_images	= args.n;		// typically will be 64 (batch size)

	// calculate the number of images this thread needs to load at once
	// e.g., 64 batch size / 6 threads = 10 or 11 images per thread
	args.n = (idx + 1) * number_of_images / number_of_threads - idx * number_of_images / number_of_threads;

	while (image_data_loading_threads_must_exit == false and
			cfg_and_state.must_immediately_exit == false)
	{
		/// @todo get rid of this busy-loop

		// wait until the control thread tells us we can load the next set of images
		if (data_loading_per_thread_flag[idx] == 0)
		{
			Darknet::TimingAndTracking tat2(name, false, "SLEEPING!");
			std::this_thread::sleep_for(thread_wait_ms);
			continue;
		}

		// if we get here, then the control thread has told us to load the next images

		args_swap_mutex.lock();
		load_args args_local = args_swap[idx];
		args_swap_mutex.unlock();

		Darknet::load_single_image_data(args_local);

		data_loading_per_thread_flag[idx] = 0;
	}

	cfg_and_state.del_thread_name();

	return;
}


void Darknet::run_image_loading_control_thread(load_args args)
{
	TAT(TATPARMS);

	/* NOTE:  This is normally started on a new thread!  For example, you might see this:
	 *
	 *		std::thread t(Darknet::run_image_loading_control_thread, args);
	 *
	 * There is a new one of these threads started to run this function at *every* iteration!
	 */

	const std::string name = "image loading control thread";
	cfg_and_state.set_thread_name(name);

	const auto timestamp1 = std::chrono::high_resolution_clock::now();

	if (args.threads == 0)
	{
		args.threads = 1;
	}
	const int number_of_threads	= args.threads;	// typically will be 6
	const int number_of_images	= args.n;		// typically will be 64 (batch size)

	data * out = args.d;
	data * buffers = (data*)xcalloc(number_of_threads, sizeof(data));

	// create the secondary threads (this should only happen once)
	if (data_loading_threads.empty())
	{
		*cfg_and_state.output << "Creating " << number_of_threads << " permanent CPU threads to load images and bounding boxes." << std::endl;

		data_loading_threads			.reserve(number_of_threads);
		data_loading_per_thread_flag	.reserve(number_of_threads);

		args_swap = (load_args *)xcalloc(number_of_threads, sizeof(load_args));

		for (int idx = 0; idx < number_of_threads; ++idx)
		{
			data_loading_per_thread_flag.push_back(0);
			data_loading_threads.emplace_back(image_loading_loop, idx, args);
		}
	}

	// tell each thread that we want more images, and where they can be stored
	for (int idx = 0; idx < number_of_threads; ++idx)
	{
		args.d = buffers + idx;
		args.n = (idx + 1) * number_of_images / number_of_threads - idx * number_of_images / number_of_threads;

		args_swap_mutex.lock();
		args_swap[idx] = args;
		args_swap_mutex.unlock();

		data_loading_per_thread_flag[idx] = 1;
	}

	// wait for the loading threads to be done
	for (int idx = 0; idx < number_of_threads; ++idx)
	{
		while (image_data_loading_threads_must_exit == false and
				cfg_and_state.must_immediately_exit == false and
				data_loading_per_thread_flag[idx] != 0) // the loading thread will reset this flag to zero once it is ready
		{
			Darknet::TimingAndTracking tat2(name, false, "SLEEPING!");
			std::this_thread::sleep_for(thread_wait_ms);
		}
	}

	// process the results
	*out = concat_datas(buffers, number_of_threads);
	out->shallow = 0;

	for (int idx = 0; idx < number_of_threads; ++idx)
	{
		buffers[idx].shallow = 1;
		Darknet::free_data(buffers[idx]);
	}
	free(buffers);

	const auto timestamp2 = std::chrono::high_resolution_clock::now();
	out->nanoseconds_to_load = std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp2 - timestamp1).count();

	cfg_and_state.del_thread_name();

	return;
}


void Darknet::stop_image_loading_threads()
{
	TAT(TATPARMS);

	if (not data_loading_threads.empty())
	{
		image_data_loading_threads_must_exit = true;

		for (auto & t : data_loading_threads)
		{
			if (t.joinable())
			{
				t.join();
			}
		}
		free(args_swap);
		data_loading_threads.clear();

		image_data_loading_threads_must_exit = false;
	}

	return;
}


matrix concat_matrix(matrix m1, matrix m2)
{
	TAT(TATPARMS);

	int i, count = 0;
	matrix m;
	m.cols = m1.cols;
	m.rows = m1.rows+m2.rows;
	m.vals = (float**)xcalloc(m1.rows + m2.rows, sizeof(float*));
	for(i = 0; i < m1.rows; ++i)
	{
		m.vals[count++] = m1.vals[i];
	}
	for(i = 0; i < m2.rows; ++i)
	{
		m.vals[count++] = m2.vals[i];
	}
	return m;
}


data concat_data(data d1, data d2)
{
	TAT(TATPARMS);

	data d = {0};
	d.shallow = 1;
	d.X = concat_matrix(d1.X, d2.X);
	d.y = concat_matrix(d1.y, d2.y);

	return d;
}


void get_next_batch(data d, int n, int offset, float *X, float *y)
{
	TAT(TATPARMS);

	int j;
	for(j = 0; j < n; ++j)
	{
		int index = offset + j;
		memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
		memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
	}
}


data get_data_part(data d, int part, int total)
{
	TAT(TATPARMS);

	data p = {0};
	p.shallow = 1;
	p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
	p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
	p.X.cols = d.X.cols;
	p.y.cols = d.y.cols;
	p.X.vals = d.X.vals + d.X.rows * part / total;
	p.y.vals = d.y.vals + d.y.rows * part / total;

	return p;
}
