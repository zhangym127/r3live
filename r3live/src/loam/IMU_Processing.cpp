#include "IMU_Processing.hpp"
#define COV_OMEGA_NOISE_DIAG 1e-1
#define COV_ACC_NOISE_DIAG 0.4
#define COV_GYRO_NOISE_DIAG 0.2

#define COV_BIAS_ACC_NOISE_DIAG 0.05
#define COV_BIAS_GYRO_NOISE_DIAG 0.1

#define COV_START_ACC_DIAG 1e-1
#define COV_START_GYRO_DIAG 1e-1
// #define COV_NOISE_EXT_I2C_R (0.0 * 1e-3)
// #define COV_NOISE_EXT_I2C_T (0.0 * 1e-3)
// #define COV_NOISE_EXT_I2C_Td (0.0 * 1e-3)



double g_lidar_star_tim = 0;
ImuProcess::ImuProcess() : b_first_frame_( true ), imu_need_init_( true ), last_imu_( nullptr ), start_timestamp_( -1 )
{
    Eigen::Quaterniond q( 0, 1, 0, 0 );
    Eigen::Vector3d    t( 0, 0, 0 );
    init_iter_num = 1;
    cov_acc = Eigen::Vector3d( COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG );
    cov_gyr = Eigen::Vector3d( COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG );
    mean_acc = Eigen::Vector3d( 0, 0, -9.805 );
    mean_gyr = Eigen::Vector3d( 0, 0, 0 );
    angvel_last = Zero3d;
    cov_proc_noise = Eigen::Matrix< double, DIM_OF_PROC_N, 1 >::Zero();
    // Lidar_offset_to_IMU = Eigen::Vector3d(0.0, 0.0, -0.0);
    // fout.open(DEBUG_FILE_DIR("imu.txt"),std::ios::out);
}

ImuProcess::~ImuProcess()
{ /**fout.close();*/
}

void ImuProcess::Reset()
{
    ROS_WARN( "Reset ImuProcess" );
    angvel_last = Zero3d;
    cov_proc_noise = Eigen::Matrix< double, DIM_OF_PROC_N, 1 >::Zero();

    cov_acc = Eigen::Vector3d( COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG );
    cov_gyr = Eigen::Vector3d( COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG );
    mean_acc = Eigen::Vector3d( 0, 0, -9.805 );
    mean_gyr = Eigen::Vector3d( 0, 0, 0 );

    imu_need_init_ = true;
    b_first_frame_ = true;
    init_iter_num = 1;

    last_imu_ = nullptr;

    // gyr_int_.Reset(-1, nullptr);
    start_timestamp_ = -1;
    v_imu_.clear();
    IMU_pose.clear();

    cur_pcl_un_.reset( new PointCloudXYZINormal() );
}

void ImuProcess::IMU_Initial( const MeasureGroup &meas, StatesGroup &state_inout, int &N )
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/
    ROS_INFO( "IMU Initializing: %.1f %%", double( N ) / MAX_INI_COUNT * 100 );
    Eigen::Vector3d cur_acc, cur_gyr;

    if ( b_first_frame_ )
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
    }

    for ( const auto &imu : meas.imu )
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += ( cur_acc - mean_acc ) / N;
        mean_gyr += ( cur_gyr - mean_gyr ) / N;

        cov_acc = cov_acc * ( N - 1.0 ) / N + ( cur_acc - mean_acc ).cwiseProduct( cur_acc - mean_acc ) * ( N - 1.0 ) / ( N * N );
        cov_gyr = cov_gyr * ( N - 1.0 ) / N + ( cur_gyr - mean_gyr ).cwiseProduct( cur_gyr - mean_gyr ) * ( N - 1.0 ) / ( N * N );
        // cov_acc = Eigen::Vector3d(0.1, 0.1, 0.1);
        // cov_gyr = Eigen::Vector3d(0.01, 0.01, 0.01);
        N++;
    }

    // TODO: fix the cov
    cov_acc = Eigen::Vector3d( COV_START_ACC_DIAG, COV_START_ACC_DIAG, COV_START_ACC_DIAG );
    cov_gyr = Eigen::Vector3d( COV_START_GYRO_DIAG, COV_START_GYRO_DIAG, COV_START_GYRO_DIAG );
    state_inout.gravity = Eigen::Vector3d( 0, 0, 9.805 );
    state_inout.rot_end = Eye3d;
    state_inout.bias_g = mean_gyr;
}

/**
 * @brief 通过对IMU数据的积分，获得状态变量的先验，并取得协方差矩阵
 * 
 * @param meas[in] 传感器测量数据
 * @param state_inout[inout] 状态变量 
 */
void ImuProcess::lic_state_propagate( const MeasureGroup &meas, StatesGroup &state_inout )
{
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu; //取到当前点云帧对应的所有IMU，放在v_imu中
    v_imu.push_front( last_imu_ ); //把上一个点云帧对应的最后一帧IMU放到v_imu的最前面

    // const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();  //IMU数据的结束时间
    const double &pcl_beg_time = meas.lidar_beg_time;                 //当前点云帧的起始时间

    /* 对点云中的点按照时间戳顺序排序 */
    /*** sort point clouds by offset time ***/
    PointCloudXYZINormal pcl_out = *( meas.lidar );
    std::sort( pcl_out.points.begin(), pcl_out.points.end(), time_list );
    const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double( 1000 );
    double        end_pose_dt = pcl_end_time - imu_end_time; //计算出点云的最后一个点与IMU的最后一帧之间的时间戳，这应该是很小的一个值

    /* IMU积分，号称是预积分，但是除了协方差递推，并未见到预积分相关的实施 */
    state_inout = imu_preintegration( state_inout, v_imu, end_pose_dt );
    last_imu_ = meas.imu.back();
}

/**
 * @brief 检查当前状态，避免异常输入状态
 * 
 *  主要是检查当前的速度，如果速度超过10米每秒（即36km/h）则认为当前速度是异常速度，清零异常的速度
 *  
 * @param state_inout 当前状态
 * @return true 
 * @return false 
 */
// Avoid abnormal state input
bool check_state( StatesGroup &state_inout )
{
    bool is_fail = false;
    for ( int idx = 0; idx < 3; idx++ )
    {
        if ( fabs( state_inout.vel_end( idx ) ) > 10 )
        {
            is_fail = true;
            scope_color( ANSI_COLOR_RED_BG );
            for ( int i = 0; i < 10; i++ )
            {
                cout << __FILE__ << ", " << __LINE__ << ", check_state fail !!!! " << state_inout.vel_end.transpose() << endl;
            }
            state_inout.vel_end( idx ) = 0.0;
        }
    }
    return is_fail;
}

// Avoid abnormal state input
void check_in_out_state( const StatesGroup &state_in, StatesGroup &state_inout )
{
    if ( ( state_in.pos_end - state_inout.pos_end ).norm() > 1.0 )
    {
        scope_color( ANSI_COLOR_RED_BG );
        for ( int i = 0; i < 10; i++ )
        {
            cout << __FILE__ << ", " << __LINE__ << ", check_in_out_state fail !!!! " << state_in.pos_end.transpose() << " | "
                 << state_inout.pos_end.transpose() << endl;
        }
        state_inout.pos_end = state_in.pos_end;
    }
}

std::mutex g_imu_premutex;

/**
 * @brief IMU预积分
 * 
 * 实际上只干了两件事：
 * 1. 对IMU进行积分，获得位姿先验，更新状态变量
 * 2. 计算出协方差矩阵
 * 
 * 该方法在时间上非常精确，严格从上一次更新的截至处开始当前周期的更新，结束点也要精确计算当前帧点云的截至时间和IMU数据的截至时间差
 * 
 * @param state_in[inout] 当前的EKF状态
 * @param v_imu [in] 当前帧点云对应的所有IMU数据，以及前一帧点云的最后一个IMU数据
 * @param end_pose_dt [in] 当前帧点云的截止时间戳和IMU数据截止时间戳之差
 * @return StatesGroup 返回更新后的状态变量，包括协方差矩阵
 */
StatesGroup ImuProcess::imu_preintegration( const StatesGroup &state_in, std::deque< sensor_msgs::Imu::ConstPtr > &v_imu, double end_pose_dt )
{
    std::unique_lock< std::mutex > lock( g_imu_premutex );
    StatesGroup                    state_inout = state_in;

    /* 检查当前状态，速度超过10m/s则将对应的速度清零 */
    if ( check_state( state_inout ) )
    {
        state_inout.display( state_inout, "state_inout" );
        state_in.display( state_in, "state_in" );
    }

    Eigen::Vector3d acc_imu( 0, 0, 0 ), angvel_avr( 0, 0, 0 ), acc_avr( 0, 0, 0 ), vel_imu( 0, 0, 0 ), pos_imu( 0, 0, 0 );
    vel_imu = state_inout.vel_end;
    pos_imu = state_inout.pos_end;
    Eigen::Matrix3d R_imu( state_inout.rot_end );
    Eigen::MatrixXd F_x( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Identity() );
    Eigen::MatrixXd cov_w( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Zero() );
    double          dt = 0;
    int             if_first_imu = 1;
    // printf("IMU start_time = %.5f, end_time = %.5f, state_update_time = %.5f, start_delta = %.5f\r\n", v_imu.front()->header.stamp.toSec() -
    // g_lidar_star_tim,
    //        v_imu.back()->header.stamp.toSec() - g_lidar_star_tim,
    //        state_in.last_update_time - g_lidar_star_tim,
    //        state_in.last_update_time - v_imu.front()->header.stamp.toSec());

    /* 遍历IMU数据 */
    for ( std::deque< sensor_msgs::Imu::ConstPtr >::iterator it_imu = v_imu.begin(); it_imu != ( v_imu.end() - 1 ); it_imu++ )
    {
        // if(g_lidar_star_tim == 0 || state_inout.last_update_time == 0)
        // {
        //   return state_inout;
        // }

        /* 同时取两帧IMU */
        sensor_msgs::Imu::ConstPtr head = *( it_imu );
        sensor_msgs::Imu::ConstPtr tail = *( it_imu + 1 );

        /* 取两帧IMU的数据的均值 */
        angvel_avr << 0.5 * ( head->angular_velocity.x + tail->angular_velocity.x ), 0.5 * ( head->angular_velocity.y + tail->angular_velocity.y ),
            0.5 * ( head->angular_velocity.z + tail->angular_velocity.z );
        acc_avr << 0.5 * ( head->linear_acceleration.x + tail->linear_acceleration.x ),
            0.5 * ( head->linear_acceleration.y + tail->linear_acceleration.y ), 0.5 * ( head->linear_acceleration.z + tail->linear_acceleration.z );

        /* 剔除角速度偏差 */
        angvel_avr -= state_inout.bias_g;
        /* 剔除加速度偏差 */
        acc_avr = acc_avr - state_inout.bias_a;

        /* 从上一次的更新结束点开始，从这一点可以看出对时间的精确性要求很高 */
        if ( tail->header.stamp.toSec() < state_inout.last_update_time )
        {
            continue;
        }

        /* 获得相邻两帧IMU的时间差 */
        if ( if_first_imu )
        {
            if_first_imu = 0;
            dt = tail->header.stamp.toSec() - state_inout.last_update_time;
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }
        if ( dt > 0.05 )
        {
            dt = 0.05;
        }

        /* 计算状态协方差 */
        /* covariance propagation */
        Eigen::Matrix3d acc_avr_skew;
        Eigen::Matrix3d Exp_f = Exp( angvel_avr, dt );
        acc_avr_skew << SKEW_SYM_MATRIX( acc_avr );
        // Eigen::Matrix3d Jr_omega_dt = right_jacobian_of_rotion_matrix<double>(angvel_avr*dt);
        Eigen::Matrix3d Jr_omega_dt = Eigen::Matrix3d::Identity();
        F_x.block< 3, 3 >( 0, 0 ) = Exp_f.transpose();
        // F_x.block<3, 3>(0, 9) = -Eye3d * dt;
        F_x.block< 3, 3 >( 0, 9 ) = -Jr_omega_dt * dt;
        // F_x.block<3,3>(3,0)  = -R_imu * off_vel_skew * dt;
        F_x.block< 3, 3 >( 3, 3 ) = Eye3d; // Already the identity.
        F_x.block< 3, 3 >( 3, 6 ) = Eye3d * dt;
        F_x.block< 3, 3 >( 6, 0 ) = -R_imu * acc_avr_skew * dt;
        F_x.block< 3, 3 >( 6, 12 ) = -R_imu * dt;
        F_x.block< 3, 3 >( 6, 15 ) = Eye3d * dt;

        Eigen::Matrix3d cov_acc_diag, cov_gyr_diag, cov_omega_diag;
        cov_omega_diag = Eigen::Vector3d( COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG, COV_OMEGA_NOISE_DIAG ).asDiagonal();
        cov_acc_diag = Eigen::Vector3d( COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG, COV_ACC_NOISE_DIAG ).asDiagonal();
        cov_gyr_diag = Eigen::Vector3d( COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG, COV_GYRO_NOISE_DIAG ).asDiagonal();
        // cov_w.block<3, 3>(0, 0) = cov_omega_diag * dt * dt;
        cov_w.block< 3, 3 >( 0, 0 ) = Jr_omega_dt * cov_omega_diag * Jr_omega_dt * dt * dt;
        cov_w.block< 3, 3 >( 3, 3 ) = R_imu * cov_gyr_diag * R_imu.transpose() * dt * dt;
        cov_w.block< 3, 3 >( 6, 6 ) = cov_acc_diag * dt * dt;
        cov_w.block< 3, 3 >( 9, 9 ).diagonal() =
            Eigen::Vector3d( COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG, COV_BIAS_GYRO_NOISE_DIAG ) * dt * dt; // bias gyro covariance
        cov_w.block< 3, 3 >( 12, 12 ).diagonal() =
            Eigen::Vector3d( COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG, COV_BIAS_ACC_NOISE_DIAG ) * dt * dt; // bias acc covariance

        // cov_w.block<3, 3>(18, 18).diagonal() = Eigen::Vector3d(COV_NOISE_EXT_I2C_R, COV_NOISE_EXT_I2C_R, COV_NOISE_EXT_I2C_R) * dt * dt; // bias
        // gyro covariance cov_w.block<3, 3>(21, 21).diagonal() = Eigen::Vector3d(COV_NOISE_EXT_I2C_T, COV_NOISE_EXT_I2C_T, COV_NOISE_EXT_I2C_T) * dt
        // * dt;  // bias acc covariance cov_w(24, 24) = COV_NOISE_EXT_I2C_Td * dt * dt;

        /* 协方差递推 */
        state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

        /* 姿态、加速度、位置、速度积分 */
        R_imu = R_imu * Exp_f;
        acc_imu = R_imu * acc_avr - state_inout.gravity;
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
        vel_imu = vel_imu + acc_imu * dt;
        angvel_last = angvel_avr;
        acc_s_last = acc_imu;

        // cout <<  std::setprecision(3) << " dt = " << dt << ", acc: " << acc_avr.transpose()
        //      << " acc_imu: " << acc_imu.transpose()
        //      << " vel_imu: " << vel_imu.transpose()
        //      << " omega: " << angvel_avr.transpose()
        //      << " pos_imu: " << pos_imu.transpose()
        //       << endl;
        // cout << "Acc_avr: " << acc_avr.transpose() << endl;
    }

    /* 下面将积分的结果更新到状态变量，同时考虑最后一个IMU数据和点云最后一个点的时间差，把位姿都补偿进去 */

    // cout <<__FILE__ << ", " << __LINE__ <<" ,diagnose lio_state = " << std::setprecision(2) <<(state_inout - StatesGroup()).transpose() << endl;
    /*** calculated the pos and attitude prediction at the frame-end ***/
    dt = end_pose_dt;

    state_inout.last_update_time = v_imu.back()->header.stamp.toSec() + dt;
    // cout << "Last update time = " <<  state_inout.last_update_time - g_lidar_star_tim << endl;
    if ( dt > 0.1 )
    {
        scope_color( ANSI_COLOR_RED_BOLD );
        for ( int i = 0; i < 1; i++ )
        {
            cout << __FILE__ << ", " << __LINE__ << "dt = " << dt << endl;
        }
        dt = 0.1;
    }
    state_inout.vel_end = vel_imu + acc_imu * dt;
    state_inout.rot_end = R_imu * Exp( angvel_avr, dt );
    state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    // cout <<__FILE__ << ", " << __LINE__ <<" ,diagnose lio_state = " << std::setprecision(2) <<(state_inout - StatesGroup()).transpose() << endl;

    // cout << "Preintegration State diff = " << std::setprecision(2) << (state_inout - state_in).head<15>().transpose()
    //      <<  endl;
    // std::cout << __FILE__ << " " << __LINE__ << std::endl;
    // check_state(state_inout);
    if ( 0 )
    {
        if ( check_state( state_inout ) )
        {
            // printf_line;
            std::cout << __FILE__ << " " << __LINE__ << std::endl;
            state_inout.display( state_inout, "state_inout" );
            state_in.display( state_in, "state_in" );
        }
        check_in_out_state( state_in, state_inout );
    }
    // cout << (state_inout - state_in).transpose() << endl;
    return state_inout;
}

/**
 * @brief IMU积分，然后对当前帧点云去畸变
 * 
 * 对IMU进行积分，基于积分的结果完成对点云的去畸变校正
 * 
 * 对IMU的积分和imu_preintegration基本类似，但是积分的结果仅用于点云的去畸变校正，并未实质性的更新状态变量
 * 
 * 与FAST-LIO2中的UndistortPcl主要区别在于取消了使用EKF对IMU进行积分并获得状态变量估计值的方式，
 * 而是直接根据运动学方程基于上一帧IMU的状态计算当前帧IMU的状态，获得位姿估计值。
 * 其他部分几乎完全一致，实现上更简单了。
 * 
 * IMU积分除了计算姿态和位置之外，还计算了速度和加速度，计算加速度的时候剔除了重力加速度，不知道重力加速度的
 * 方向是如何处置的。计算出来的状态变量用于点云的帧内校正，并没有输出，否则就和imu_preintegration重复了。
 * 
 * 点云的去畸变部分除了是反向（即从最后一个点开始）之外，没有什么特别的。
 * 
 * @param meas[in] 包含IMU、点云的原始测量数据
 * @param _state_inout[inout] 当前的状态变量 
 * @param pcl_out [out] 校正后点云
 */
void ImuProcess::lic_point_cloud_undistort( const MeasureGroup &meas, const StatesGroup &_state_inout, PointCloudXYZINormal &pcl_out )
{
    StatesGroup state_inout = _state_inout;
    auto        v_imu = meas.imu;
    v_imu.push_front( last_imu_ );
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;
    /*** sort point clouds by offset time ***/
    pcl_out = *( meas.lidar );
    std::sort( pcl_out.points.begin(), pcl_out.points.end(), time_list );
    const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double( 1000 );
    /*std::cout << "[ IMU Process ]: Process lidar from " << pcl_beg_time - g_lidar_star_tim << " to " << pcl_end_time- g_lidar_star_tim << ", "
              << meas.imu.size() << " imu msgs from " << imu_beg_time- g_lidar_star_tim << " to " << imu_end_time- g_lidar_star_tim
              << ", last tim: " << state_inout.last_update_time- g_lidar_star_tim << std::endl;
    */

    /* 用上一周期的IMU数据以及EKF的当前状态state_inout初始化IMU位姿 */
    /*** Initialize IMU pose ***/
    IMU_pose.clear();
    // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
    IMU_pose.push_back( set_pose6d( 0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end ) );

    /* 正向传播，对当前点云帧对应的IMU数据进行位姿推算，推算出每一帧IMU对应的位姿，保存在IMUpose中 */    
    /*** forward propagation at each imu point ***/
    Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu( state_inout.vel_end ), pos_imu( state_inout.pos_end );
    Eigen::Matrix3d R_imu( state_inout.rot_end );
    Eigen::MatrixXd F_x( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Identity() );
    Eigen::MatrixXd cov_w( Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES >::Zero() );
    double          dt = 0;

    /* 遍历每一帧IMU数据 */
    for ( auto it_imu = v_imu.begin(); it_imu != ( v_imu.end() - 1 ); it_imu++ )
    {
        /* 每次取出两帧IMU数据 */
        auto &&head = *( it_imu );
        auto &&tail = *( it_imu + 1 );

        /* 取两帧IMU角速度和加速度的平均值 */
        angvel_avr << 0.5 * ( head->angular_velocity.x + tail->angular_velocity.x ), 0.5 * ( head->angular_velocity.y + tail->angular_velocity.y ),
            0.5 * ( head->angular_velocity.z + tail->angular_velocity.z );
        acc_avr << 0.5 * ( head->linear_acceleration.x + tail->linear_acceleration.x ),
            0.5 * ( head->linear_acceleration.y + tail->linear_acceleration.y ), 0.5 * ( head->linear_acceleration.z + tail->linear_acceleration.z );

        /* 消除角速度和加速度偏差 */
        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr - state_inout.bias_a;

#ifdef DEBUG_PRINT
// fout<<head->header.stamp.toSec()<<" "<<angvel_avr.transpose()<<" "<<acc_avr.transpose()<<std::endl;
#endif
        /* 取得当前帧IMU和上一帧IMU的时间差dt */
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        /* covariance propagation */

        Eigen::Matrix3d acc_avr_skew;
        Eigen::Matrix3d Exp_f = Exp( angvel_avr, dt ); // 根据角速度和dt计算当前帧IMU的旋转增量
        acc_avr_skew << SKEW_SYM_MATRIX( acc_avr );
#ifdef DEBUG_PRINT
// fout<<head->header.stamp.toSec()<<" "<<angvel_avr.transpose()<<" "<<acc_avr.transpose()<<std::endl;
#endif

        /* 更新姿态 */
        /* propagation of IMU attitude */
        R_imu = R_imu * Exp_f;

        /* 更新加速度，转到地图坐标系，剔除重力加速度 */
        /* Specific acceleration (global frame) of IMU */
        acc_imu = R_imu * acc_avr - state_inout.gravity;

        /* 更新位置 */
        /* propagation of IMU */
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

        /* 更新速度 */
        /* velocity of IMU */
        vel_imu = vel_imu + acc_imu * dt;

        /* save the poses at each IMU measurements */
        angvel_last = angvel_avr;
        acc_s_last = acc_imu;

        /* 计算当前IMU帧相对于当前点云帧起始时间的偏移量 */
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        // std::cout<<"acc "<<acc_imu.transpose()<<"vel "<<acc_imu.transpose()<<"vel "<<pos_imu.transpose()<<std::endl;

        /* 记录当前IMU帧对应的位姿数据，下面用来对雷达进行帧内校正 */
        IMU_pose.push_back( set_pose6d( offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu ) );
    }

    /* 当前点云帧对应的IMU的积分已经完成，用积分的结果更新EKF的状态state_inout */
    /*** calculated the pos and attitude prediction at the frame-end ***/
    dt = pcl_end_time - imu_end_time;
    state_inout.vel_end = vel_imu + acc_imu * dt;
    state_inout.rot_end = R_imu * Exp( angvel_avr, dt );
    state_inout.pos_end = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    Eigen::Vector3d pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lidar_offset_to_IMU;
    // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

#ifdef DEBUG_PRINT
    std::cout << "[ IMU Process ]: vel " << state_inout.vel_end.transpose() << " pos " << state_inout.pos_end.transpose() << " ba"
              << state_inout.bias_a.transpose() << " bg " << state_inout.bias_g.transpose() << std::endl;
    std::cout << "propagated cov: " << state_inout.cov.diagonal().transpose() << std::endl;
#endif

    /* 下面开始用IMU积分结果进行点云帧内校正，反向遍历正向传播获得的所有位姿 */
    /*** undistort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1;
    for ( auto it_kp = IMU_pose.end() - 1; it_kp != IMU_pose.begin(); it_kp-- )
    {
        auto head = it_kp - 1;
        R_imu << MAT_FROM_ARRAY( head->rot );           //当前帧IMU对应的旋转
        acc_imu << VEC_FROM_ARRAY( head->acc );         //当前帧IMU对应的加速度
        // std::cout<<"head imu acc: "<<acc_imu.transpose()<<std::endl;
        vel_imu << VEC_FROM_ARRAY( head->vel );         //当前帧IMU对应的速度
        pos_imu << VEC_FROM_ARRAY( head->pos );         //当前帧IMU对应的位移
        angvel_avr << VEC_FROM_ARRAY( head->gyr );      //当前帧IMU对应的角速度

        /* 按照倒序，对点云中时间戳大于当前帧IMU时间戳的点进行校正 */
        for ( ; it_pcl->curvature / double( 1000 ) > head->offset_time; it_pcl-- )
        {
            /* 取得从当前点到当前帧IMU的时差 */
            dt = it_pcl->curvature / double( 1000 ) - head->offset_time;

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            // 取得当前点对应的世界坐标系姿态，R_imu记录的当前帧IMU对应的世界坐标系姿态
            Eigen::Matrix3d R_i( R_imu * Exp( angvel_avr, dt ) );   
            // 取得当前点对应的世界坐标系位置，pos_imu记录的是当前帧IMU对应的世界坐标系位置
            Eigen::Vector3d T_ei( pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_offset_to_IMU - pos_liD_e ); 

            // 取得待校正的点坐标
            Eigen::Vector3d P_i( it_pcl->x, it_pcl->y, it_pcl->z );
            /**
             * 对当前点进行校正：
             * 1. 首先将当前点变换到对应的世界坐标系：( R_i * P_i + T_ei )
             * 2. 然后用当前帧点云（最后一个点）对应的世界坐标系位姿，减去当前点对应的世界坐标系位姿，获得所有点相对于最后一点的相对位姿*/ 
            Eigen::Vector3d P_compensate = state_inout.rot_end.transpose() * ( R_i * P_i + T_ei );

            /// save Undistorted points and their rotation
            it_pcl->x = P_compensate( 0 );
            it_pcl->y = P_compensate( 1 );
            it_pcl->z = P_compensate( 2 );

            if ( it_pcl == pcl_out.points.begin() )
                break;
        }
    }
}

void ImuProcess::Process( const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZINormal::Ptr cur_pcl_un_ )
{
    // double t1, t2, t3;
    // t1 = omp_get_wtime();

    if ( meas.imu.empty() )
    {
        // std::cout << "no imu data" << std::endl;
        return;
    };
    ROS_ASSERT( meas.lidar != nullptr );

    if ( imu_need_init_ )
    {
        /// The very first lidar frame
        IMU_Initial( meas, stat, init_iter_num );

        imu_need_init_ = true;

        last_imu_ = meas.imu.back();

        if ( init_iter_num > MAX_INI_COUNT )
        {
            imu_need_init_ = false;
            // std::cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<std::endl;
            ROS_INFO(
                "IMU Initials: Gravity: %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",
                stat.gravity[ 0 ], stat.gravity[ 1 ], stat.gravity[ 2 ], stat.bias_g[ 0 ], stat.bias_g[ 1 ], stat.bias_g[ 2 ], cov_acc[ 0 ],
                cov_acc[ 1 ], cov_acc[ 2 ], cov_gyr[ 0 ], cov_gyr[ 1 ], cov_gyr[ 2 ] );
        }

        return;
    }

    /// Undistort points： the first point is assummed as the base frame
    /// Compensate lidar points with IMU rotation (with only rotation now)
    // if ( 0 || (stat.last_update_time < 0.1))

    if ( 0 )
    {
        // UndistortPcl(meas, stat, *cur_pcl_un_);
    }
    else
    {
        if ( 1 )
        {
            /* 进行点云的帧内校正 */
            lic_point_cloud_undistort( meas, stat, *cur_pcl_un_ );
        }
        else
        {
            *cur_pcl_un_ = *meas.lidar;
        }
        /* 积分IMU数据，获得状态变量的先验值以及协方差矩阵 */
        lic_state_propagate( meas, stat );
    }
    // t2 = omp_get_wtime();

    last_imu_ = meas.imu.back();

    // t3 = omp_get_wtime();

    // std::cout<<"[ IMU Process ]: Time: "<<t3 - t1<<std::endl;
}