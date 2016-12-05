#include "ros/ros.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;

Mat rgbImg;
bool rgbInit;

float x_clip_min;
float x_clip_max;
float y_clip_min;
float y_clip_max;

tf::TransformListener *tf_listener;

void saveRGBImg(const sensor_msgs::ImageConstPtr& msg)
{
    try {
        rgbImg = cv_bridge::toCvCopy(msg, "bgr8")->image;
        rgbInit = true;
    } catch(cv_bridge::Exception& e) {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void findBlocks(const sensor_msgs::PointCloud2ConstPtr& sensor_message_pc)
{
    if (!rgbInit) return;

    tf::StampedTransform transform;
    ros::Time now = ros::Time::now();
    tf_listener->waitForTransform ("/world", "/kinect2_rgb_optical_frame", now, ros::Duration(4.0));
    tf_listener->lookupTransform ("/world", "/kinect2_rgb_optical_frame", now, transform);

    pcl::PointCloud<pcl::PointXYZ> sensor_pc;
    pcl::fromROSMsg(*sensor_message_pc, sensor_pc);
    pcl::PointCloud<pcl::PointXYZ> world_pc;

    // transform from camera frame to world frame
    sensor_pc.header.frame_id = "/kinect2_rgb_optical_frame";
    pcl_ros::transformPointCloud("/world", sensor_pc, world_pc, *tf_listener);

    // crop the image based on x and y clipping values
    int pxMin = 0, pxMax = 0, pyMin = 0, pyMax = 0;
    for (int i = 0; i < 1920; i++) {

        pcl::PointXYZ pt = world_pc[1036800 + i]; // use middle column
        if (isnan(pt.x)) continue;

        // x is increasing in opposite direction from array access
        if (pxMin == 0 && pt.x < x_clip_max) {
            pxMin = i;
        }
        if (pxMax == 0 && pt.x < x_clip_min) {
            pxMax = i;
            break;
        }
    }

    for (int i = 0; i < 2073600; i += 1920) {

        pcl::PointXYZ pt = world_pc[960 + i]; // use middle row
        if (isnan(pt.y)) continue;

        if (pyMin == 0 && pt.y > y_clip_min) {
            pyMin = i / 1920;
        }
        if (pyMax == 0 && pt.y > y_clip_max) {
            pyMax = i / 1920;
            break;
        }
    }

    //cout << "image bounds: " << pxMin << ", " << pxMax << ", " << pyMin << ", " << pyMax << endl;

    cv::Rect rect(pxMin, pyMin, pxMax - pxMin, pyMax - pyMin);
    cv::Mat croppedImg;
    croppedImg = rgbImg(rect);

    Mat src_gray;
    int thresh = 100;
    int max_thresh = 255;

    cvtColor(croppedImg, src_gray, CV_BGR2GRAY);
    blur(src_gray, src_gray, Size(3,3));
    cv::imshow("view", src_gray);

    RNG rng(12345);
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<RotatedRect> minRect( contours.size() );
    vector<Moments> mu( contours.size() );
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ ) {
        /// Get the moments
        mu[i] = moments( contours[i], false );

        /// Get the rotated rectangles
        minRect[i] = minAreaRect( Mat(contours[i]) );

        ///  Get the mass centers:
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ ) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        circle( drawing, mc[i], 4, color, -1, 8, 0 );

        // rotated rectangle
        Point2f rect_points[4]; minRect[i].points( rect_points );
        for( int j = 0; j < 4; j++ )
        line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

        float rotation = minRect[i].angle;
        ostringstream ss;
        ss << rotation;
        string r(ss.str());
        putText(drawing, r.c_str(), cvPoint(mc[i].x,mc[i].y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
    }

    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "block_detection_listener");
    ros::NodeHandle* n = new ros::NodeHandle();
    tf::TransformListener tfl;
    tf_listener = &tfl;

    rgbInit = false;

    // TODO: put this in launch file to share with filter_server.cpp (pc filter)
    /*n->getParam("x_clip_min", x_clip_min_);
    n->getParam("x_clip_max", x_clip_max_);
    n->getParam("y_clip_min", y_clip_min_);
    n->getParam("y_clip_max", y_clip_max_);*/

    x_clip_min = -0.18;
    x_clip_max = 0.6;
    y_clip_min = -0.65;
    y_clip_max = -0.15;

    ros::Subscriber original_pc_sub = n->subscribe("/kinect2/hd/points", 1, findBlocks);

    cv::namedWindow("view");
    cv::startWindowThread();
    image_transport::ImageTransport it(*n);
    image_transport::Subscriber sub = it.subscribe("/kinect2/hd/image_color_rect", 1, saveRGBImg);

    ros::spin();
    return 0;
}
