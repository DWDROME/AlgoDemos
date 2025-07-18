#ifndef MY_PLANNER_H
#define MY_PLANNER_H

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>

namespace my_planner {
    class MyPlanner : public nav_core::BaseLocalPlanner {
        public:
            MyPlanner();
            ~MyPlanner();

            void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);
            bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);
            bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
            bool isGoalReached();

        private:
            

    };

}


#endif
