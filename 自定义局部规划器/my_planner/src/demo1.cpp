#include "my_planner.h"
#include <pluginlib/class_list_macros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

PLUGINLIB_EXPORT_CLASS(my_planner::MyPlanner, nav_core::BaseLocalPlanner)


namespace my_planner{

    MyPlanner::MyPlanner(){
        setlocale(LC_ALL,""); //字符串编码本地化
    }
    MyPlanner::~MyPlanner(){

    }

    tf::TransformListener* tf_listener_;
    costmap_2d::Costmap2DROS* costmap_ros_;
    void MyPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros){
        ROS_WARN("启动局部规划器");
        tf_listener_ = new tf::TransformListener(); // 初始化tf监听器
        costmap_ros_ = costmap_ros; // 保存costmap_ros指针
    }

    std::vector<geometry_msgs::PoseStamped> global_plan_; // 全局路径
    int target_index_ = 0; // 目标点索引
    bool pose_adjusting_;
    bool goal_reached_ ;
    bool MyPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan){
        target_index_ = 0; // 重置目标点索引
        global_plan_ = plan; // 保存全局路径
        pose_adjusting_ = false;
        goal_reached_ = false; 
        return true; // 返回true表示设置路径成功
    }

    bool MyPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel){
        // 获取代价地图数据

        costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
        unsigned char* map_data = costmap->getCharMap();
        unsigned int size_x = costmap->getSizeInCellsX();
        unsigned int size_y = costmap->getSizeInCellsY(); 

        cv::Mat map_image(size_y, size_x, CV_8UC3, cv::Scalar(128,128,128)); // 创建灰色图像
        for(unsigned int y = 0; y < size_y; y++){
            for(unsigned int x = 0; x < size_x; x++){
                int map_index = y*size_x +x;
                unsigned char cost = map_data[map_index];
                cv::Vec3b& pixel = map_image.at<cv::Vec3b>(map_index);
                if(cost ==0){  //可通行区域
                    pixel = cv::Vec3b(128, 128, 128); // 灰色
                }
                else if (cost == 254){  //障碍物
                    pixel = cv::Vec3b(0,0,0); // 黑色
                }
                else if (cost == 253){  // 禁行区域
                    pixel = cv::Vec3b(255, 255, 0); // 浅蓝色
                }
                else{
                    unsigned char blue = 255-cost;
                    unsigned char red  = cost;
                    pixel = cv::Vec3b(blue, 0, red); // 蓝色到红色渐变
                }
            }
        }

        for(int i=0;i<global_plan_.size();i++){
            geometry_msgs::PoseStamped pose_odom;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("tianracer/odom", global_plan_[i], pose_odom);
            double odom_x = pose_odom.pose.position.x;
            double odom_y = pose_odom.pose.position.y;
            
            double origin_x = costmap->getOriginX();
            double origin_y = costmap->getOriginY();
            double local_x = odom_x - origin_x; // 计算局部坐标系下的x坐标
            double local_y = odom_y - origin_y; // 计算局部坐标
            int x = local_x / costmap->getResolution(); // 将局部坐标转换为像素坐标
            int y = local_y / costmap->getResolution(); // 将局部坐标
            cv::circle(map_image, cv::Point(x, y), 0, cv::Scalar(255, 0, 255)); // 绘制全局路径点
            
            if(i >= target_index_ && i< target_index_ +10){
                cv::circle(map_image, cv::Point(x, y), 0, cv::Scalar(0, 255, 255)); // 检测路径点
                int map_index = y* size_x +x;
                unsigned char cost = map_data[map_index];
                if(cost>=253){
                    return false;
                }
            }
        }    
        

        map_image.at<cv::Vec3b>(size_y/2, size_x/2) = cv::Vec3b(0, 255, 0); // 绘制机器人当前位置
 
        cv::Mat flipped_image(size_x,size_y, CV_8UC3,cv::Scalar(128,128,128));
        for(unsigned int y = 0; y < size_y; y++){
            for(unsigned int x = 0; x < size_x; x++){
                cv::Vec3b pixel = map_image.at<cv::Vec3b>(y, x);
                flipped_image.at<cv::Vec3b>((size_x - 1 - x),(size_y - 1 - y)) = pixel; // 翻转图像
            }
        }
        map_image = flipped_image; // 更新地图图像
        // 显示代价地图
        cv::namedWindow("Map");
        cv::resize(map_image, map_image, cv::Size(size_y*5, size_x*5), 0, 0, cv::INTER_NEAREST);
        cv::resizeWindow("Map",size_y*5, size_x*5);
        cv::imshow("Map", map_image);

        int final_index = global_plan_.size() - 1;
        geometry_msgs::PoseStamped pose_final;
        global_plan_[final_index].header.stamp = ros::Time(0);
        tf_listener_->transformPose("tianracer/base_link", global_plan_[final_index], pose_final);
        
        if(!pose_adjusting_){
            double dx = pose_final.pose.position.x;
            double dy = pose_final.pose.position.y;
            double dist = std::sqrt(dx*dx + dy*dy);
            if(dist < 0.05){
                pose_adjusting_ = true;
            }
        }else{
            double final_yaw = tf::getYaw(pose_final.pose.orientation);
            ROS_WARN("调整最终姿态，final_yaw = %.2f", final_yaw);
            cmd_vel.linear.x = pose_final.pose.position.x * 1.5;
            cmd_vel.angular.z = final_yaw * 5.0; // 调整朝向
            if(abs(final_yaw)<0.1){
                goal_reached_ = true;
                ROS_WARN("到达终点！");
                cmd_vel.linear.x = 0.0; // 停止前进
                cmd_vel.angular.z = 0.0; // 停止转向
            }
        }

        geometry_msgs::PoseStamped target_pose;
        for(int i = target_index_;i<global_plan_.size();i++){
            geometry_msgs::PoseStamped pose_base;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("tianracer/base_link",global_plan_[i],pose_base);
            double dx = pose_base.pose.position.x;
            double dy = pose_base.pose.position.y;
            double dist = std::sqrt(dx*dx + dy*dy);
            if(dist > 0.2){
                target_pose = pose_base; // 找到下一个目标点
                target_index_ = i;
                ROS_WARN("选择第 %d 个路径点为临时目标，距离= %.2f ",target_index_, dist);
                break;
            }

            if(i == final_index){
                ROS_WARN("已到达全局路径终点，停止规划");
                target_index_ = i; // 到达终点
                return false; // 返回false表示没有速度命令
            }
        }
        cmd_vel.linear.x = target_pose.pose.position.x * 1.5; // 设置线速度
        cmd_vel.angular.z = target_pose.pose.position.y * 5.0; // 设置


        // cmd_vel.linear.x = target_pose.pose.position.x * 1.5; // 设置线速度
        // cmd_vel.linear.y = target_pose.pose.position.y * 1.5; // 设置线速度
        
        // geometry_msgs::PoseStamped pose_base, pose_map;
        // pose_map.pose.position.x = 1.0;
        // pose_map.pose.position.y = -1.5;
        // pose_map.pose.orientation.w = 1.0;
        // pose_map.header.frame_id = "map";
        // pose_map.header.stamp = ros::Time(0);
        // tf_listener_->transformPose("tianracer/base_link", pose_map,pose_base);
        // cmd_vel.angular.z = pose_base.pose.position.y * 1.0; // 设置

        cv::Mat plan_image(600,600,CV_8UC3, cv::Scalar(0,0,0));

        for(int i=0;i<global_plan_.size();i++){
            geometry_msgs::PoseStamped  pose_base;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("tianracer/base_link", global_plan_[i], pose_base);
            int cv_x = 300 - pose_base.pose.position.x*100;
            int cv_y = 300 - pose_base.pose.position.y*100;
            cv::circle(plan_image, cv::Point(cv_x,cv_y),1,cv::Scalar(255,0,255));
            
        }

        cv::circle(plan_image, cv::Point(300,300), 15, cv::Scalar(0,255,0)); // 绘制机器人当前位置
        cv::line(plan_image, cv::Point(65,300), cv::Point(510,300), cv::Scalar(0,255,0), 1); // 绘制机器人朝向
        cv::line(plan_image, cv::Point(300,45), cv::Point(300,555), cv::Scalar(0,255,0), 1); // 绘制机器人朝向
        
        // cv::namedWindow("Plan");
        // cv::imshow("Plan",plan_image);


        cv::waitKey(1);

        return true; // 返回true表示计算速度命令成功
    }
    bool MyPlanner::isGoalReached(){
        return goal_reached_; // 返回是否到达目标点
    }

}