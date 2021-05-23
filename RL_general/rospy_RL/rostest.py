import numpy as np
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import GetModelState

class jointStates():
    def __init__(self):
        self.jointP = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])

s = jointStates()
def subCB(data):                                                                                                                  #copied from kuka_teleop.py, may need editing
    s.jointP=np.array(list(data.position))
def get_jt(jtp_list,t):
    jtp_list = jtp_list + s.jointP
    jt = JointTrajectory()
    jt.joint_names = ["iiwa_joint_1","iiwa_joint_2","iiwa_joint_3","iiwa_joint_4","iiwa_joint_5","iiwa_joint_6","iiwa_joint_7"]
    jtp = JointTrajectoryPoint()
    jtp.positions = jtp_list
    jtp.time_from_start = rospy.Duration.from_sec(t)
    jt.points.append(jtp)
    return jt
pubJoint = rospy.Publisher('/iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=10)                                            #publish ball position???????idk
subJoint = rospy.Subscriber("/iiwa/joint_states", JointState, subCB)
link_ros = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
model_ros = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
rospy.init_node('iiwa_joints', anonymous=True)
# rospy.spin()
print(link_ros('ball', 'world'))
# jtp = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
# jt = get_jt(jtp,1)
# jt = np.array(jt.points[0].positions)
# print(jt)
print(s.jointP)
