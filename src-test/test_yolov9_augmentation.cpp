#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "data.hpp"
#include "darknet_internal.hpp"

namespace
{
	std::filesystem::path make_temp_dir(const std::string & name)
	{
		const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
		const auto dir = std::filesystem::temp_directory_path() / (name + "_" + std::to_string(stamp));
		std::filesystem::create_directories(dir);
		return dir;
	}

	void write_image_and_label(const std::filesystem::path & image_path, const cv::Mat & image, const std::string & label)
	{
		ASSERT_TRUE(cv::imwrite(image_path.string(), image));
		std::ofstream labels(image_path.parent_path() / (image_path.stem().string() + ".txt"));
		ASSERT_TRUE(labels.good());
		labels << label << "\n";
	}

	load_args make_yolov9_args(std::vector<std::string> & paths, data & loaded, const int w, const int h, const int max_boxes = 16)
	{
		static std::vector<char *> path_ptrs;
		path_ptrs.clear();
		for (auto & path : paths)
		{
			path_ptrs.push_back(path.data());
		}

		load_args args = {0};
		args.paths = path_ptrs.data();
		args.n = 1;
		args.m = static_cast<int>(paths.size());
		args.w = w;
		args.h = h;
		args.c = 3;
		args.classes = 1;
		args.num_boxes = max_boxes;
		args.truth_size = 6;
		args.mini_batch = 1;
		args.augment_speed = 1;
		args.augment_policy = 1;
		args.degrees = 0.0f;
		args.translate = 0.0f;
		args.yolov9_scale = 0.0f;
		args.shear = 0.0f;
		args.perspective = 0.0f;
		args.hsv_h = 0.0f;
		args.hsv_s = 0.0f;
		args.hsv_v = 0.0f;
		args.flipud_prob = 0.0f;
		args.fliplr_prob = 0.0f;
		args.mosaic_prob = 0.0f;
		args.mixup_prob = 0.0f;
		args.copy_paste_prob = 0.0f;
		args.d = &loaded;
		args.type = DETECTION_DATA;
		return args;
	}

	int count_truth_boxes(const data & loaded, const int truth_size, const int max_boxes)
	{
		int count = 0;
		for (int i = 0; i < max_boxes; ++i)
		{
			if (loaded.y.vals[0][i * truth_size] > 0.0f)
			{
				++count;
			}
		}
		return count;
	}
}

TEST(YOLOv9Augmentation, LetterboxPathKeepsBoxOnlyLabelInReferenceCoordinates)
{
	const auto dir = make_temp_dir("darknet_yolov9_letterbox");
	const auto image_path = dir / "wide.png";
	cv::Mat image(2, 4, CV_8UC3, cv::Scalar(10, 20, 30));
	write_image_and_label(image_path, image, "0 0.5 0.5 0.5 1.0");

	std::vector<std::string> paths = {image_path.string()};
	data loaded = {0};
	load_args args = make_yolov9_args(paths, loaded, 8, 8);

	Darknet::load_single_image_data(args);

	ASSERT_EQ(1, loaded.X.rows);
	ASSERT_EQ(8 * 8 * 3, loaded.X.cols);
	ASSERT_NE(nullptr, loaded.X.vals[0]);
	ASSERT_NEAR(0.5f, loaded.y.vals[0][0], 1.0e-6f);
	ASSERT_NEAR(0.5f, loaded.y.vals[0][1], 1.0e-6f);
	ASSERT_NEAR(0.5f, loaded.y.vals[0][2], 1.0e-6f);
	ASSERT_NEAR(0.5f, loaded.y.vals[0][3], 1.0e-6f);
	ASSERT_NEAR(0.0f, loaded.y.vals[0][4], 1.0e-6f);

	Darknet::free_data(loaded);
	std::filesystem::remove_all(dir);
}

TEST(YOLOv9Augmentation, FlipProbabilityUpdatesNormalizedLabel)
{
	const auto dir = make_temp_dir("darknet_yolov9_flip");
	const auto image_path = dir / "flip.png";
	cv::Mat image(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
	write_image_and_label(image_path, image, "0 0.3 0.5 0.4 0.5");

	std::vector<std::string> paths = {image_path.string()};
	data loaded = {0};
	load_args args = make_yolov9_args(paths, loaded, 8, 8);
	args.fliplr_prob = 1.0f;

	Darknet::load_single_image_data(args);

	ASSERT_NEAR(0.7f, loaded.y.vals[0][0], 1.0e-6f);
	ASSERT_NEAR(0.5f, loaded.y.vals[0][1], 1.0e-6f);
	ASSERT_NEAR(0.4f, loaded.y.vals[0][2], 1.0e-6f);
	ASSERT_NEAR(0.5f, loaded.y.vals[0][3], 1.0e-6f);

	Darknet::free_data(loaded);
	std::filesystem::remove_all(dir);
}

TEST(YOLOv9Augmentation, BoxOnlyCopyPasteDoesNotDuplicateMosaicLabels)
{
	const auto dir = make_temp_dir("darknet_yolov9_mosaic");
	std::vector<std::string> paths;
	for (int i = 0; i < 4; ++i)
	{
		const auto image_path = dir / ("mosaic_" + std::to_string(i) + ".png");
		cv::Mat image(8, 8, CV_8UC3, cv::Scalar(10 + i, 20 + i, 30 + i));
		write_image_and_label(image_path, image, "0 0.5 0.5 1.0 1.0");
		paths.push_back(image_path.string());
	}

	data loaded = {0};
	load_args args = make_yolov9_args(paths, loaded, 8, 8);
	args.mosaic_prob = 1.0f;
	args.copy_paste_prob = 1.0f;

	Darknet::load_single_image_data(args);

	const int boxes = count_truth_boxes(loaded, args.truth_size, args.num_boxes);
	ASSERT_GT(boxes, 0);
	ASSERT_LE(boxes, 4);

	Darknet::free_data(loaded);
	std::filesystem::remove_all(dir);
}
