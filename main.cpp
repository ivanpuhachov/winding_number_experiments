#include <igl/fast_winding_number.h>
#include <igl/read_triangle_mesh.h>
#include <igl/slice_mask.h>
#include <Eigen/Geometry>
#include <igl/octree.h>
#include <igl/barycenter.h>
#include <igl/knn.h>
#include <igl/random_points_on_mesh.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/per_face_normals.h>
#include <igl/copyleft/cgal/point_areas.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/get_seconds.h>
#include <iostream>
#include <cstdlib>
#include <igl/point_mesh_squared_distance.h>
#include <igl/parula.h>


int main(int argc, char *argv[])
{
    const auto time = [](std::function<void(void)> func)->double
    {
        const double t_before = igl::get_seconds();
        func();
        const double t_after = igl::get_seconds();
        return t_after-t_before;
    };

    std::cout << "Reading file" << std::endl;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
//    igl::read_triangle_mesh("../bunny.off",V,F);
    igl::read_triangle_mesh("../bathtub.obj", V,F);

    // Normalize to unit sphere
    std::cout << "Normalizing dimensions" << std::endl;
    const Eigen::RowVector3d minDims = V.colwise().minCoeff();
    V = V - Eigen::VectorXd::Ones(V.rows()) * minDims;
    const double maxDims = V.maxCoeff();
//    V = V.array() / (Eigen::MatrixXd::Ones(V.rows(), V.cols()) * maxDims).array();
    V = V.array() / maxDims;
    const double maxDimX = V.col(0).maxCoeff();
    const double maxDimY = V.col(1).maxCoeff();
    const double maxDimZ = V.col(2).maxCoeff();
    V = V.array() * 0.9 + 0.05;


    // Sample mesh for point cloud
    std::cout << "Sampling points on faces" << std::endl;
    Eigen::MatrixXd P,N;
    {
        Eigen::VectorXi I;
        Eigen::SparseMatrix<double> B;
        igl::random_points_on_mesh(10000,V,F,B,I);
        P = B*V;
        Eigen::MatrixXd FN;
        igl::per_face_normals(V,F,FN);
        N.resize(P.rows(),3);
        for(int p = 0;p<I.rows();p++)
        {
            N.row(p) = FN.row(I(p));
        }
    }

    // Query points in the bounding box
    std::cout << "Generating query points" << std::endl;
    Eigen::MatrixXd Q(1000000, 3);
    for (int i_x=0; i_x<100; ++i_x)
    {
        for (int i_y=0; i_y<100; ++i_y)
        {
            Eigen::MatrixXd myblock (100,3);
            myblock.col(2) = Eigen::ArrayXd::LinSpaced(100,0.005,0.995) * maxDimZ;
            myblock.col(1) = Eigen::VectorXd::Ones(100) * (0.005 + i_y * 0.01) * maxDimY;
            myblock.col(0) = Eigen::VectorXd::Ones(100) * (0.005 + i_x * 0.01) * maxDimX;
            Q.block(100*i_y + 10000*i_x, 0, 100, 3) = myblock;
        }
    }

    std::cout << "Getting Distance Function at query points" << std::endl;
    Eigen::VectorXd sqrD;
    Eigen::VectorXi II;
    Eigen::MatrixXd CC;
    igl::point_mesh_squared_distance(Q,V,F,sqrD,II,CC);
    Eigen::VectorXd Q_df = sqrD.array().sqrt();

    std::cout << "Evaluating sign of DF" << std::endl;


    // Positions of points inside of triangle soup (V,F)
    Eigen::MatrixXd QiV;
    Eigen::VectorXf WiV;

    igl::FastWindingNumberBVH fwn_bvh;
    printf("triangle soup precomputation    (% 8ld triangles): %g secs\n",
           F.rows(),
           time([&](){igl::fast_winding_number(V.cast<float>().eval(),F,2,fwn_bvh);}));

    printf("      triangle soup evaluation  (% 8ld queries):   %g secs\n",
           Q.rows(),
           time([&](){igl::fast_winding_number(fwn_bvh,2,Q.cast<float>().eval(),WiV);}));
    igl::slice_mask(Q,WiV.array()>0.5,1,QiV);


    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V,F);
    int query_data = 0;
    viewer.data_list[query_data].set_mesh(V,F);
    viewer.data_list[query_data].clear();
    viewer.data_list[query_data].point_size = 6;
    viewer.append_mesh();
    int object_data = 1;
    viewer.data_list[object_data].set_mesh(V,F);
    viewer.data_list[object_data].point_size = 5;

//    Eigen::MatrixXd Colors(Q.rows(), Q.cols());
//    igl::parula(WiV, false, Colors);
//    viewer.data_list[query_data].set_points(Q, Colors);

    int i_row = 0;
    auto update = [Q, WiV, sqrD, Q_df, query_data](igl::opengl::glfw::Viewer &viewer, int ni_row){
        ni_row = ni_row % 100;
        Eigen::MatrixXd mblock = Q.block(ni_row*10000, 0, 10000, 3);
        Eigen::VectorXd mblock_wn = WiV.block(ni_row*10000, 0, 10000, 3).cast<double>();
        Eigen::VectorXd mblock_df = sqrD.block(ni_row*10000, 0, 10000, 1);
        Eigen::VectorXd mblock_sdf = ((mblock_wn.array()>0.5).cast<double>()*2 - 1) * mblock_df.array();
        Eigen::MatrixXd Colors(10000, 3);
        Eigen::VectorXd visvector = mblock_wn;
        igl::parula(visvector, false, Colors);
        viewer.data_list[query_data].set_points(mblock,Colors);
        std::cout << "Row " << ni_row << ", max: " << visvector.maxCoeff() << ", min" << visvector.minCoeff() << std::endl;
    };


    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
            {
                switch(key)
                {
                    default:
                        return false;
                    case '1':
                        i_row = (i_row + 100 - 1) % 100;
                        break;
                    case '2':
                        i_row = (i_row + 1) % 100;
                        break;
                }
                update(viewer, i_row);
                return true;
            };

    update(viewer, i_row);
    viewer.launch();

}