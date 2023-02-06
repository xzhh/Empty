//
// Created by jn98zk on 6/25/22.
//

#include "CuboidStructureCalc.hpp"

namespace Geometry {



    CuboidStructureCalc::CuboidStructureCalc(const coordinate_type &gridSize) : cuboid_size(gridSize) {
        cuboid_location = CuboidPair{Coordinate3D<int>{0,0,0},gridSize};
        initialize_self();
    }

    CuboidStructureCalc::CuboidStructureCalc(const CuboidPair &cuboidLocation) : cuboid_location(cuboidLocation) {
        cuboid_size = cuboidLocation.upper_right - cuboidLocation.lower_left;
        initialize_self();
    }

    void CuboidStructureCalc::initialize_self() {
        cuboid_items = CuboidStructureCalc::calculate_cuboid_attributes(cuboid_location);
        cuboid_frame_extension = CuboidStructureCalc::calculate_cuboid_frame_extension(cuboid_items,1);

    }

    std::array<CuboidPair, 38> CuboidStructureCalc::calculate_cuboid_attributes(const CuboidPair &cuboid_location) {
        const auto cuboid_dimensions = cuboid_location.cuboid_size();
        std::array<CuboidPair, 38> cuboid_attributes{};
        using coord = CuboidPair::coordinate_type;
        /// Corners
        auto cuboid_corners = calculate_cuboid_corners(cuboid_location);
        {
            uint8_t corner_index{};
            auto constexpr corner_extender = Coordinate3D<int>{1, 1, 1};
            for (uint8_t out_index = Geometry::BOTTOM_LOWER_LEFT_CORNER;
                 out_index <= Geometry::TOP_UPPER_RIGHT_CORNER; ++out_index, ++corner_index) {
                auto const &cuboid_corner = cuboid_corners[corner_index];
                cuboid_attributes[out_index] = {cuboid_corner, cuboid_corner + corner_extender};
            }
        }
        const auto [x_dist, y_dist, z_dist] = cuboid_dimensions;
        const auto &bottom_lower_left_corner = cuboid_attributes[Geometry::BOTTOM_LOWER_LEFT_CORNER].lower_left;
        const auto &bottom_upper_left_corner = cuboid_attributes[Geometry::BOTTOM_UPPER_LEFT_CORNER].lower_left;
        const auto &bottom_lower_right_corner = cuboid_attributes[Geometry::BOTTOM_LOWER_RIGHT_CORNER].lower_left;
        const auto &top_lower_left_corner = cuboid_attributes[Geometry::TOP_LOWER_LEFT_CORNER].lower_left;

        /// Bottom and Top Face
        {

            cuboid_attributes[Geometry::BOTTOM_FACE] = CuboidPair{bottom_lower_left_corner,
                                                                  bottom_lower_left_corner +
                                                                  coord{x_dist,
                                                                        y_dist, 1}};
            cuboid_attributes[Geometry::BOTTOM_LOWER_EDGE] = CuboidPair{bottom_lower_left_corner,
                                                                        bottom_lower_left_corner +
                                                                        coord{x_dist,
                                                                              1, 1}};
            cuboid_attributes[Geometry::BOTTOM_LEFT_EDGE] = CuboidPair{bottom_lower_left_corner,
                                                                       bottom_lower_left_corner +
                                                                       coord{1,
                                                                             y_dist, 1}};

            cuboid_attributes[Geometry::BOTTOM_UPPER_EDGE] = CuboidPair{bottom_upper_left_corner,
                                                                        bottom_upper_left_corner +
                                                                        coord{x_dist, 1, 1}};

            cuboid_attributes[Geometry::BOTTOM_RIGHT_EDGE] = CuboidPair{bottom_lower_right_corner,
                                                                        bottom_lower_right_corner +
                                                                        coord{1,
                                                                              y_dist, 1}};

            const auto up_displacement = coord{0, 0, z_dist - 1};

            cuboid_attributes[Geometry::TOP_FACE] = cuboid_attributes[Geometry::BOTTOM_FACE].move_by(
                    up_displacement);
            cuboid_attributes[Geometry::TOP_LOWER_EDGE] = cuboid_attributes[Geometry::BOTTOM_LOWER_EDGE].move_by(
                    up_displacement);
            cuboid_attributes[Geometry::TOP_LEFT_EDGE] = cuboid_attributes[Geometry::BOTTOM_LEFT_EDGE].move_by(
                    up_displacement);

            cuboid_attributes[Geometry::TOP_UPPER_EDGE] = cuboid_attributes[Geometry::BOTTOM_UPPER_EDGE].move_by(
                    up_displacement);

            cuboid_attributes[Geometry::TOP_RIGHT_EDGE] = cuboid_attributes[Geometry::BOTTOM_RIGHT_EDGE].move_by(
                    up_displacement);

        }
        /// Left and Right Face
        {
            cuboid_attributes[Geometry::LEFT_FACE] = CuboidPair{bottom_lower_left_corner,
                                                                bottom_lower_left_corner +
                                                                coord{1, y_dist, z_dist}};
            cuboid_attributes[Geometry::LEFT_LOWER_EDGE] = CuboidPair{bottom_lower_left_corner,
                                                                      bottom_lower_left_corner +
                                                                      coord{1, y_dist, 1}};
            cuboid_attributes[Geometry::LEFT_LEFT_EDGE] = CuboidPair{bottom_lower_left_corner,
                                                                     bottom_lower_left_corner +
                                                                     coord{1, 1, z_dist}};
            cuboid_attributes[Geometry::LEFT_UPPER_EDGE] = CuboidPair{top_lower_left_corner, top_lower_left_corner +
                                                                                             coord{1,
                                                                                                   y_dist,
                                                                                                   1}};
            cuboid_attributes[Geometry::LEFT_RIGHT_EDGE] = CuboidPair{bottom_upper_left_corner,
                                                                      bottom_upper_left_corner +
                                                                      coord{1, 1,
                                                                            z_dist}};

            const auto right_displacement = coord{x_dist - 1, 0, 0};

            cuboid_attributes[Geometry::RIGHT_FACE] = cuboid_attributes[Geometry::LEFT_FACE].move_by(
                    right_displacement);
            cuboid_attributes[Geometry::RIGHT_LOWER_EDGE] = cuboid_attributes[Geometry::LEFT_LOWER_EDGE].move_by(
                    right_displacement);
            cuboid_attributes[Geometry::RIGHT_LEFT_EDGE] = cuboid_attributes[Geometry::LEFT_LEFT_EDGE].move_by(
                    right_displacement);

            cuboid_attributes[Geometry::RIGHT_UPPER_EDGE] = cuboid_attributes[Geometry::LEFT_UPPER_EDGE].move_by(
                    right_displacement);

            cuboid_attributes[Geometry::RIGHT_RIGHT_EDGE] = cuboid_attributes[Geometry::LEFT_RIGHT_EDGE].move_by(
                    right_displacement);

        }
        /// Front and Back Face
        {
            cuboid_attributes[Geometry::BACK_FACE] = CuboidPair{bottom_lower_left_corner,
                                                                bottom_lower_left_corner +
                                                                coord{x_dist, 1, z_dist}};
            cuboid_attributes[Geometry::BACK_LOWER_EDGE] = CuboidPair{bottom_lower_left_corner,
                                                                      bottom_lower_left_corner +
                                                                      coord{x_dist, 1, 1}};
            cuboid_attributes[Geometry::BACK_LEFT_EDGE] = CuboidPair{bottom_lower_left_corner,
                                                                     bottom_lower_left_corner +
                                                                     coord{1, 1, z_dist}};
            cuboid_attributes[Geometry::BACK_UPPER_EDGE] = CuboidPair{top_lower_left_corner,
                                                                      top_lower_left_corner + coord{x_dist, 1, 1}};
            cuboid_attributes[Geometry::BACK_RIGHT_EDGE] = CuboidPair{bottom_lower_right_corner,
                                                                      bottom_lower_right_corner +
                                                                      coord{1, 1, z_dist}};

            const auto inside_displacement = coord{0, y_dist - 1, 0};

            cuboid_attributes[Geometry::FRONT_FACE] = cuboid_attributes[Geometry::BACK_FACE].move_by(
                    inside_displacement);
            cuboid_attributes[Geometry::FRONT_LOWER_EDGE] = cuboid_attributes[Geometry::BACK_LOWER_EDGE].move_by(
                    inside_displacement);
            cuboid_attributes[Geometry::FRONT_LEFT_EDGE] = cuboid_attributes[Geometry::BACK_LEFT_EDGE].move_by(
                    inside_displacement);

            cuboid_attributes[Geometry::FRONT_UPPER_EDGE] = cuboid_attributes[Geometry::BACK_UPPER_EDGE].move_by(
                    inside_displacement);

            cuboid_attributes[Geometry::FRONT_RIGHT_EDGE] = cuboid_attributes[Geometry::BACK_RIGHT_EDGE].move_by(
                    inside_displacement);

        }

        return cuboid_attributes;
    }

    std::array<Coordinate3D<int>, 8> CuboidStructureCalc::calculate_cuboid_corners(const CuboidPair &cuboid_location) {
        std::array<coordinate_type, 8> anchor_positions{};
        const auto cuboid_dimensions = cuboid_location.cuboid_size();
        anchor_positions[0] = cuboid_location.lower_left;
        anchor_positions[1] = cuboid_location.lower_left + coordinate_type{cuboid_dimensions.x - 1, 0, 0};
        anchor_positions[2] = cuboid_location.lower_left + coordinate_type{0, cuboid_dimensions.y - 1, 0};
        anchor_positions[3] =
                cuboid_location.lower_left + coordinate_type{cuboid_dimensions.x - 1, cuboid_dimensions.y - 1, 0};
        anchor_positions[4] = cuboid_location.lower_left + coordinate_type{0, 0, cuboid_dimensions.z - 1};
        anchor_positions[5] =
                cuboid_location.lower_left + coordinate_type{cuboid_dimensions.x - 1, 0, cuboid_dimensions.z - 1};
        anchor_positions[6] =
                cuboid_location.lower_left + coordinate_type{0, cuboid_dimensions.y - 1, cuboid_dimensions.z - 1};
        anchor_positions[7] = cuboid_location.upper_right - coordinate_type{1, 1, 1};
        return anchor_positions;
    }

    const CuboidPair &CuboidStructureCalc::getCuboidLocation() const {
        return cuboid_location;
    }

    const CuboidStructureCalc::coordinate_type &CuboidStructureCalc::getCuboidSize() const {
        return cuboid_size;
    }

    const std::array<CuboidPair, 38> &CuboidStructureCalc::getCuboidItems() const {
        return cuboid_items;
    }

    const std::array<CuboidPair, 38> &CuboidStructureCalc::getCuboidFrameExtension() const {
        return cuboid_frame_extension;
    }

    std::array<CuboidPair, 38>
    CuboidStructureCalc::calculate_cuboid_frame_extension(const std::array<CuboidPair, 38> &cuboid_attributes,
                                                          int frame_size) {
        if (frame_size!=1){
            throw std::runtime_error("Only Frame == 1 supported");
        }
        std::array<CuboidPair, 38> extension_positions{};
        extension_positions[BOTTOM_FACE] = cuboid_attributes[BOTTOM_FACE].move_by({0, 0, -1});
        extension_positions[BOTTOM_LOWER_EDGE] = cuboid_attributes[BOTTOM_LOWER_EDGE].move_by({0, -1, -1});
        extension_positions[BOTTOM_LEFT_EDGE] = cuboid_attributes[BOTTOM_LEFT_EDGE].move_by({-1, 0, -1});
        extension_positions[BOTTOM_UPPER_EDGE] = cuboid_attributes[BOTTOM_UPPER_EDGE].move_by({0, 1, -1});
        extension_positions[BOTTOM_RIGHT_EDGE] = cuboid_attributes[BOTTOM_RIGHT_EDGE].move_by({1, 0, -1});
        extension_positions[TOP_FACE] = cuboid_attributes[TOP_FACE].move_by({0, 0, 1});
        extension_positions[TOP_LOWER_EDGE] = cuboid_attributes[TOP_LOWER_EDGE].move_by({0, -1, 1});
        extension_positions[TOP_LEFT_EDGE] = cuboid_attributes[TOP_LEFT_EDGE].move_by({-1, 0, 1});
        extension_positions[TOP_UPPER_EDGE] = cuboid_attributes[TOP_UPPER_EDGE].move_by({0, 1, 1});
        extension_positions[TOP_RIGHT_EDGE] = cuboid_attributes[TOP_RIGHT_EDGE].move_by({1, 0, 1});
        extension_positions[LEFT_FACE] = cuboid_attributes[LEFT_FACE].move_by({-1, 0, 0});
        extension_positions[LEFT_LOWER_EDGE] = cuboid_attributes[LEFT_LOWER_EDGE].move_by({-1, 0, -1});
        extension_positions[LEFT_LEFT_EDGE] = cuboid_attributes[LEFT_LEFT_EDGE].move_by({-1, -1, 0});
        extension_positions[LEFT_UPPER_EDGE] = cuboid_attributes[LEFT_UPPER_EDGE].move_by({-1, 0, 1});
        extension_positions[LEFT_RIGHT_EDGE] = cuboid_attributes[LEFT_RIGHT_EDGE].move_by({-1, 1, 0});
        extension_positions[RIGHT_FACE] = cuboid_attributes[RIGHT_FACE].move_by({1, 0, 0});
        extension_positions[RIGHT_LOWER_EDGE] = cuboid_attributes[RIGHT_LOWER_EDGE].move_by({1, 0, -1});
        extension_positions[RIGHT_LEFT_EDGE] = cuboid_attributes[RIGHT_LEFT_EDGE].move_by({1, -1, 0});
        extension_positions[RIGHT_UPPER_EDGE] = cuboid_attributes[RIGHT_UPPER_EDGE].move_by({1, 0, 1});
        extension_positions[RIGHT_RIGHT_EDGE] = cuboid_attributes[RIGHT_RIGHT_EDGE].move_by({1, 1, 0});
        extension_positions[FRONT_FACE] = cuboid_attributes[FRONT_FACE].move_by({0, 1, 0});
        extension_positions[FRONT_LOWER_EDGE] = cuboid_attributes[FRONT_LOWER_EDGE].move_by({0, 1, -1});
        extension_positions[FRONT_LEFT_EDGE] = cuboid_attributes[FRONT_LEFT_EDGE].move_by({-1, 1, 0});
        extension_positions[FRONT_UPPER_EDGE] = cuboid_attributes[FRONT_UPPER_EDGE].move_by({0, 1, 1});
        extension_positions[FRONT_RIGHT_EDGE] = cuboid_attributes[FRONT_RIGHT_EDGE].move_by({1, 1, 0});
        extension_positions[BACK_FACE] = cuboid_attributes[BACK_FACE].move_by({0, -1, 0});
        extension_positions[BACK_LOWER_EDGE] = cuboid_attributes[BACK_LOWER_EDGE].move_by({0, -1, -1});
        extension_positions[BACK_LEFT_EDGE] = cuboid_attributes[BACK_LEFT_EDGE].move_by({-1, -1, 0});
        extension_positions[BACK_UPPER_EDGE] = cuboid_attributes[BACK_UPPER_EDGE].move_by({0, -1, 1});
        extension_positions[BACK_RIGHT_EDGE] = cuboid_attributes[BACK_RIGHT_EDGE].move_by({1, -1, 0});
        extension_positions[BOTTOM_LOWER_LEFT_CORNER] = cuboid_attributes[BOTTOM_LOWER_LEFT_CORNER].move_by(
                {-1, -1, -1});
        extension_positions[BOTTOM_LOWER_RIGHT_CORNER] = cuboid_attributes[BOTTOM_LOWER_RIGHT_CORNER].move_by(
                {1, -1, -1});
        extension_positions[BOTTOM_UPPER_LEFT_CORNER] = cuboid_attributes[BOTTOM_UPPER_LEFT_CORNER].move_by(
                {-1, 1, -1});
        extension_positions[BOTTOM_UPPER_RIGHT_CORNER] = cuboid_attributes[BOTTOM_UPPER_RIGHT_CORNER].move_by(
                {1, 1, -1});
        extension_positions[TOP_LOWER_LEFT_CORNER] = cuboid_attributes[TOP_LOWER_LEFT_CORNER].move_by({-1, -1, 1});
        extension_positions[TOP_LOWER_RIGHT_CORNER] = cuboid_attributes[TOP_LOWER_RIGHT_CORNER].move_by({1, -1, 1});
        extension_positions[TOP_UPPER_LEFT_CORNER] = cuboid_attributes[TOP_UPPER_LEFT_CORNER].move_by({-1, 1, 1});
        extension_positions[TOP_UPPER_RIGHT_CORNER] = cuboid_attributes[TOP_UPPER_RIGHT_CORNER].move_by({1, 1, 1});
        return extension_positions;
    }



} // Geometry