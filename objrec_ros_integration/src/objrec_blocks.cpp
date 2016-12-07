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

#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkTransform.h>
#include <ObjRecRANSAC/UserData.h>
#include <ObjRecRANSAC/Shapes/PointSetShape.h>
#include <objrec_ros_integration/FindObjects.h>
#include <objrec_msgs/RecognizedObjects.h>
#include <tf_conversions/tf_kdl.h>
#include <resource_retriever/retriever.h>
#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace cv;

Mat rgbImg;
bool rgbInit;
ros::NodeHandle* n;

float x_clip_min;
float x_clip_max;
float y_clip_min;
float y_clip_max;

tf::TransformListener *tf_listener;
std::string block_model_vtk;
std::string block_model_stl;
vtkPolyData *block_model_data;
UserData *block_user_data;
void load_block_model();
ros::Publisher objects_pub_;
ros::Publisher markers_pub_;

boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud;

boost::mutex buffer_mutex_;
std::list<vtkSmartPointer<vtkPolyDataReader> > readers_;

static void array_to_pose(const double* array, geometry_msgs::Pose &pose_msg)
{
    tf::Matrix3x3 rot_m =  tf::Matrix3x3(
            array[0],array[1],array[2],
            array[3],array[4],array[5],
            array[6],array[7],array[8]);
    tf::Quaternion rot_q;
    rot_m.getRotation(rot_q);
    rot_q = rot_q * tf::Quaternion(tf::Vector3(1.0,0,0), M_PI/2.0);
    tf::quaternionTFToMsg(rot_q, pose_msg.orientation);

    pose_msg.position.x = array[9];
    pose_msg.position.y = array[10];
    pose_msg.position.z = array[11];
}

void saveRGBImg(const sensor_msgs::ImageConstPtr& msg)
{
    try {
        rgbImg = cv_bridge::toCvCopy(msg, "bgr8")->image;
        rgbInit = true;
    } catch(cv_bridge::Exception& e) {
        ROS_ERROR("could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void getCloud(const sensor_msgs::PointCloud2ConstPtr &points_msg)
{
    // Lock the buffer mutex while we're capturing a new point cloud
    boost::mutex::scoped_lock buffer_lock(buffer_mutex_);

    // Convert to PCL cloud
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*points_msg, *cloud_tmp);

    cloud = cloud_tmp;
}
int findBlocks(list<boost::shared_ptr<PointSetShape> >& out)
{
    if (!rgbInit) return -1;

    tf::StampedTransform transform;
    tf::StampedTransform transform_world_to_camera;
    ros::Time now = ros::Time::now();
    tf_listener->waitForTransform ("/world", "/kinect2_rgb_optical_frame", now, ros::Duration(4.0));
    tf_listener->lookupTransform ("/world", "/kinect2_rgb_optical_frame", now, transform);
    tf_listener->lookupTransform ("/kinect2_rgb_optical_frame", "/world", now, transform_world_to_camera);
    tf::Transform tf_to_cam(transform_world_to_camera.getRotation(), transform_world_to_camera.getOrigin());

    // Lock the buffer mutex
    boost::mutex::scoped_lock buffer_lock(buffer_mutex_);

    // transform from camera frame to world frame
    pcl::PointCloud<pcl::PointXYZ> world_pc;
    cloud->header.frame_id = "/kinect2_rgb_optical_frame";
    pcl_ros::transformPointCloud("/world", *cloud, world_pc, *tf_listener);

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

    cout << "image bounds: " << pxMin << ", " << pxMax << ", " << pyMin << ", " << pyMax << endl;

    cv::Rect rect(pxMin, pyMin, pxMax - pxMin, pyMax - pyMin);
    cv::Mat croppedImg;
    croppedImg = rgbImg(rect);

    Mat src_gray;
    int thresh = 100;
    int max_thresh = 255;

    cvtColor(croppedImg, src_gray, CV_BGR2GRAY);
    blur(src_gray, src_gray, Size(3,3));

    /*double alpha = 1.05; // contrast
    int beta = 0; // brightness
    for (int y = 0; y < src_gray.rows; y++) {
        for (int x = 0; x < src_gray.cols; x++) {
            for (int c = 0; c < 3; c++) {
                //std::cout << static_cast<unsigned>(src_gray.at<Vec3b>(y,x)[c]) << ", ";// = saturate_cast<uchar>( alpha * (src_gray.at<Vec3b>(y,x)[c]) + beta);
                uchar test = saturate_cast<uchar>( alpha * (src_gray.at<Vec3b>(y,x)[c] - 20));
                //std::cout << static_cast<unsigned>(test) << ", ";

                src_gray.at<Vec3b>(y,x)[c] = test;
            }
            //std::cout << std::endl;
        }
    }*/

    RNG rng(12345);
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    std::cout << "---------- detect edges ----------" << std::endl;
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
        std::cout << contours.size() << ": " << mu[i].m00 << std::endl;
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }

    /// Draw contours
    for( int i = 0; i < contours.size(); i++ ) {
        std::cout << "---------- find orientation " << i << " ----------" << std::endl;

        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( croppedImg, contours, i, color, 2, 8, hierarchy, 0, Point() );
        circle( croppedImg, mc[i], 4, color, -1, 8, 0 );

        // rotated rectangle
        Point2f rect_points[4]; minRect[i].points( rect_points );
        for( int j = 0; j < 4; j++ )
        line( croppedImg, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );

        float rotation = minRect[i].angle;
        ostringstream ss;
        ss << rotation;
        string r(ss.str());
        putText(croppedImg, r.c_str(), cvPoint(mc[i].x,mc[i].y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

        std::cout << "---------- get pc point ----------" << std::endl;
        // index into point cloud to get position of center of mass
        // TODO: may be more accurate to find average or pc cluster
        // TODO: figure out why sometimes mc is Nan?
        if (isnan(mc[i].x) || isnan(mc[i].y)) {
            std::cout << "contour " << i << " is nan!" << std::endl;
            continue;
        }
        int px = pxMin + mc[i].x;
        int py = pyMin + mc[i].y;
        std::cout << pxMin << ", " << pyMin << ", " << mc[i].x << ", " << mc[i].y << std::endl;
        pcl::PointXYZ pt = world_pc[py*1920 + px];

        std::cout << "---------- transform to camera frame ----------" << std::endl;

        // transform point into camera frame to publish

        tf::Vector3 cam_pt;
        cam_pt.setX(pt.x);
        cam_pt.setY(pt.y);
        cam_pt.setZ(pt.z);
        cam_pt.setZ(0.01); //TODO: get proper Z value from point cloud
        cam_pt = tf_to_cam * cam_pt;

        float sinTheta = sin(rotation * KDL::deg2rad);
        float cosTheta = cos(rotation * KDL::deg2rad);
        double *rigid_transform = new double[12];
        // Rotational part
        rigid_transform[0] = cosTheta;  rigid_transform[1] = -sinTheta; rigid_transform[2] = 0;
        rigid_transform[3] = sinTheta;  rigid_transform[4] = cosTheta;  rigid_transform[5] = 0;
        rigid_transform[6] = 0;         rigid_transform[7] = 0;         rigid_transform[8] = 1;
        // The translation
        rigid_transform[9]  = cam_pt.x();
        rigid_transform[10] = cam_pt.y();
        rigid_transform[11] = cam_pt.z();

        std::cout << "---------- create new shape ----------" << std::endl;
        boost::shared_ptr<PointSetShape> shape = boost::make_shared<PointSetShape>(block_user_data, block_model_data, rigid_transform, block_model_data);
        // Save the new created shape
        out.push_back(shape);

        delete rigid_transform;
    }

    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", croppedImg );

    return 0;
}

void publish_markers(const objrec_msgs::RecognizedObjects &objects_msg)
{
    visualization_msgs::MarkerArray marker_array;
    int id = 0;

    for(std::vector<objrec_msgs::PointSetShape>::const_iterator it = objects_msg.objects.begin();
            it != objects_msg.objects.end();
            ++it)
    {
        visualization_msgs::Marker marker;

        marker.header = objects_msg.header;
        marker.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.lifetime = ros::Duration(20.0);
        marker.ns = "objrec";
        marker.id = 0;

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;

        marker.color.a = 0.75;
        marker.color.r = 1.0;
        marker.color.g = 0.1;
        marker.color.b = 0.3;

        marker.id = id++;
        marker.pose = it->pose;

        marker.mesh_resource = block_model_stl;
        marker_array.markers.push_back(marker);
    }

    // Publish the markers
    markers_pub_.publish(marker_array);
}


bool recognizeBlocks(objrec_ros_integration::FindObjects::Request &req, objrec_ros_integration::FindObjects::Response &res)
{
    std::list<boost::shared_ptr<PointSetShape> > detected_models;
    int failure = findBlocks(detected_models);
    // No objects recognized
    if (failure) {
        return false;
    }

    // Construct recognized objects message
    objrec_msgs::RecognizedObjects objects_msg;
    objects_msg.header.stamp = pcl_conversions::fromPCL(cloud->header).stamp;
    objects_msg.header.frame_id = cloud->header.frame_id;

    for(std::list<boost::shared_ptr<PointSetShape> >::iterator it = detected_models.begin();
            it != detected_models.end();
            ++it)
    {
        boost::shared_ptr<PointSetShape> detected_model = *it;

        // Construct and populate a message
        objrec_msgs::PointSetShape pss_msg;
        pss_msg.label = detected_model->getUserData()->getLabel();
        pss_msg.confidence = detected_model->getConfidence();
        array_to_pose(detected_model->getRigidTransform(), pss_msg.pose);

        // Transform into the world frame TODO: make this frame a parameter // keeping in camera frame?
        geometry_msgs::PoseStamped pose_stamped_in, pose_stamped_out;
        pose_stamped_in.header = pcl_conversions::fromPCL(cloud->header);
        pose_stamped_in.pose = pss_msg.pose;

        KDL::Frame pose;
        tf::poseMsgToKDL(pss_msg.pose, pose);
        KDL::Frame rot = KDL::Frame(KDL::Rotation::RotX(-(KDL::PI)/2.0), KDL::Vector(0,0,0));
        KDL::Frame newPose = pose * rot;
        tf::poseKDLToMsg(newPose, pss_msg.pose);

        objects_msg.objects.push_back(pss_msg);
    }

    // Publish the visualization markers
    publish_markers(objects_msg);

    // Publish the recognized objects
    objects_pub_.publish(objects_msg);

    for(std::vector<objrec_msgs::PointSetShape>::const_iterator it = objects_msg.objects.begin();
            it != objects_msg.objects.end();
            ++it)
    {
        objrec_msgs::PointSetShape object = *it;
        res.object_name.push_back(object.label);
        res.object_pose.push_back(object.pose);
    }

    ROS_INFO("sending back response ");
    return true;
}

vtkSmartPointer<vtkPolyData> scale_vtk_model(vtkSmartPointer<vtkPolyData> & m, double scale = 1.0/1000.0)
{
  vtkSmartPointer<vtkTransform> transp = vtkSmartPointer<vtkTransform>::New();
  transp->Scale(scale, scale, scale);
  vtkSmartPointer<vtkTransformPolyDataFilter> tpd = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
#if VTK_MAJOR_VERSION <= 5
  tpd->SetInput(m);
#else
  tpd->SetInputData(m);
#endif
  tpd->SetTransform(transp);
  tpd->Update();
  return tpd->GetOutput();
}

void load_block_model()
{
    ROS_INFO_STREAM("Loading block only...");

    std::string model_label = "block";

    // Get the mesh uri & store it
    std::string param_name = "model_uris/"+model_label;
    if(!n->getParam(param_name, block_model_vtk)) {
        ROS_FATAL_STREAM("Required parameter not found! Namespace: "<<n->getNamespace()<<" Parameter: "<<param_name);
        throw ros::InvalidParameterException("Parameter not found!");
    }
    param_name = "stl_uris/"+model_label;
    if(!n->getParam(param_name, block_model_stl)) {
        ROS_FATAL_STREAM("Required parameter not found! Namespace: "<<n->getNamespace()<<" Parameter: "<<param_name);
        throw ros::InvalidParameterException("Parameter not found!");
    }

    ROS_INFO_STREAM("Adding model \""<<model_label<<"\" from "<<block_model_vtk);
    // Fetch the model data with a ros resource retriever
    resource_retriever::Retriever retriever;
    resource_retriever::MemoryResource resource;

    try {
        resource = retriever.get(block_model_vtk);
    } catch (resource_retriever::Exception& e) {
        ROS_ERROR_STREAM("Failed to retrieve \""<<model_label<<"\" model file from \""<<block_model_vtk<<"\" error: "<<e.what());
        return;
    }

    // Load the model into objrec
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();
    // This copies the data from the resource structure into the polydata reader
    reader->SetBinaryInputString(
            (const char*)resource.data.get(),
            resource.size);
    reader->ReadFromInputStringOn();
    reader->Update();
    readers_.push_back(reader);

    // Get the VTK normals
    vtkSmartPointer<vtkPolyData> polydata(reader->GetOutput());
    vtkSmartPointer<vtkFloatArray> point_normals(
            vtkFloatArray::SafeDownCast(polydata->GetPointData()->GetNormals()));

    if(!point_normals) {
        ROS_ERROR_STREAM("No vertex normals for mesh: "<<block_model_vtk);
        return;
    }

    // Get the VTK points
    size_t n_points = polydata->GetNumberOfPoints();
    size_t n_normals = point_normals->GetNumberOfTuples();

    if(n_points != n_normals) {
        ROS_ERROR_STREAM("Different numbers of vertices and vertex normals for mesh: "<<block_model_vtk);
        return;
    }

    // This is just here for reference
    for(vtkIdType i = 0; i < n_points; i++)
    {
        double pV[3];
        double pN[3];

        polydata->GetPoint(i, pV);
        point_normals->GetTuple(i, pN);
    }

    // Create new model user data
    block_user_data = new UserData();
    block_user_data->setLabel(model_label.c_str());

    vtkSmartPointer<vtkPolyData> model_data = reader->GetOutput();
    block_model_data = scale_vtk_model(model_data);

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "block_detection_listener");
    n = new ros::NodeHandle();
    tf::TransformListener tfl;
    tf_listener = &tfl;

    rgbInit = false;

    // TODO: put this in launch file to share with filter_server.cpp (pc filter)
    /*n->getParam("x_clip_min", x_clip_min_);
    n->getParam("x_clip_max", x_clip_max_);
    n->getParam("y_clip_min", y_clip_min_);
    n->getParam("y_clip_max", y_clip_max_);*/

    x_clip_min = -0.18; // right bound
    x_clip_max = 0.4; // left bound
    y_clip_min = -0.65; // top bound
    y_clip_max = -0.18; // bottom bound

    ros::Subscriber original_pc_sub = n->subscribe("/kinect2/hd/points", 1, getCloud);

    cv::startWindowThread();
    image_transport::ImageTransport it(*n);
    image_transport::Subscriber sub = it.subscribe("/kinect2/hd/image_color_rect", 1, saveRGBImg);

    load_block_model();

    objects_pub_ = n->advertise<objrec_msgs::RecognizedObjects>("recognized_objects",20);
    markers_pub_ = n->advertise<visualization_msgs::MarkerArray>("recognized_objects_markers",20);

    // add FindBlocks service
    ros::ServiceServer find_blocks_server_ = n->advertiseService("find_blocks", recognizeBlocks);

    ros::spin();
    return 0;
}
