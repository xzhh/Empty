//
// Created by jn98zk on 6/25/22.
//

#ifndef DISTRIBUTEDBOXREDESIGNED_CUBOIDSTRUCTURECALC_HPP
#define DISTRIBUTEDBOXREDESIGNED_CUBOIDSTRUCTURECALC_HPP

#include <ostream>
#include "Coordinate3D.hpp"
#include "storage/FullNeighbourNodeGrid.hpp"

namespace Geometry {

    struct CuboidPair {
        using coordinate_type = Coordinate3D<int>;
        coordinate_type lower_left;
        coordinate_type upper_right;

        coordinate_type cuboid_size() const {
            return upper_right - lower_left;
        }

        bool operator==(const CuboidPair &rhs) const {
            return lower_left == rhs.lower_left &&
                   upper_right == rhs.upper_right;
        }

        bool operator!=(const CuboidPair &rhs) const {
            return !(rhs == *this);
        }

        [[nodiscard]] CuboidPair move_by(coordinate_type amount) const {
            return CuboidPair{lower_left + amount, upper_right + amount};
        }

        friend std::ostream &operator<<(std::ostream &os, const CuboidPair &pair) {
            os << "CuboidPair(" << "lower_left= " << pair.lower_left << ", upper_right= " << pair.upper_right << ")";
            return os;
        }
    };

    /// All Cuboid faces, edges and corners
    enum CuboidFeature {
        BOTTOM_FACE,
        BOTTOM_LOWER_EDGE,
        BOTTOM_LEFT_EDGE,
        BOTTOM_UPPER_EDGE,
        BOTTOM_RIGHT_EDGE,
        TOP_FACE,
        TOP_LOWER_EDGE,
        TOP_LEFT_EDGE,
        TOP_UPPER_EDGE,
        TOP_RIGHT_EDGE,
        LEFT_FACE,
        LEFT_LOWER_EDGE,
        LEFT_LEFT_EDGE,
        LEFT_UPPER_EDGE,
        LEFT_RIGHT_EDGE,
        RIGHT_FACE,
        RIGHT_LOWER_EDGE,
        RIGHT_LEFT_EDGE,
        RIGHT_UPPER_EDGE,
        RIGHT_RIGHT_EDGE,
        FRONT_FACE,
        FRONT_LOWER_EDGE,
        FRONT_LEFT_EDGE,
        FRONT_UPPER_EDGE,
        FRONT_RIGHT_EDGE,
        BACK_FACE,
        BACK_LOWER_EDGE,
        BACK_LEFT_EDGE,
        BACK_UPPER_EDGE,
        BACK_RIGHT_EDGE,
        BOTTOM_LOWER_LEFT_CORNER,
        BOTTOM_LOWER_RIGHT_CORNER,
        BOTTOM_UPPER_LEFT_CORNER,
        BOTTOM_UPPER_RIGHT_CORNER,
        TOP_LOWER_LEFT_CORNER,
        TOP_LOWER_RIGHT_CORNER,
        TOP_UPPER_LEFT_CORNER,
        TOP_UPPER_RIGHT_CORNER,
    };


    /// Maps from a neighbour relation to the corresponding feature that neighbour needs.
    /// \param relation Directional relation
    /// \return Corresponding feature
    static constexpr  CuboidFeature cuboidFacefromNeighbourRelation(John::NeighbourRelation relation) {
        using namespace John;
        switch (relation) {
            case PLANE_CENTER_LEFT:
                return LEFT_FACE;
            case PLANE_CENTER_RIGHT:
                return RIGHT_FACE;
            case PLANE_CENTER_UP:
                return FRONT_FACE;
            case PLANE_CENTER_DOWN:
                return BACK_FACE;
            case PLANE_CENTER_LEFT_UP:
                return FRONT_LEFT_EDGE;
            case PLANE_CENTER_LEFT_DOWN:
                return BACK_LEFT_EDGE;
            case PLANE_CENTER_RIGHT_UP:
                return FRONT_RIGHT_EDGE;
            case PLANE_CENTER_RIGHT_DOWN:
                return BACK_RIGHT_EDGE;
            case ABOVE_CENTER_LEFT:
                return TOP_LEFT_EDGE;
            case ABOVE_CENTER_RIGHT:
                return TOP_RIGHT_EDGE;
            case ABOVE_CENTER_UP:
                return TOP_UPPER_EDGE;
            case ABOVE_CENTER:
                return TOP_FACE;
            case ABOVE_CENTER_DOWN:
                return BACK_UPPER_EDGE;
            case ABOVE_CENTER_LEFT_UP:
                return TOP_UPPER_LEFT_CORNER;
            case ABOVE_CENTER_LEFT_DOWN:
                return TOP_LOWER_LEFT_CORNER;
            case ABOVE_CENTER_RIGHT_UP:
                return TOP_UPPER_RIGHT_CORNER;
            case ABOVE_CENTER_RIGHT_DOWN:
                return TOP_LOWER_RIGHT_CORNER;
            case BELLOW_CENTER_LEFT:
                return LEFT_LOWER_EDGE;
            case BELLOW_CENTER_RIGHT:
                return RIGHT_LOWER_EDGE;
            case BELLOW_CENTER_UP:
                return FRONT_LOWER_EDGE;
            case BELLOW_CENTER:
                return BOTTOM_FACE;
            case BELLOW_CENTER_DOWN:
                return BACK_LOWER_EDGE;
            case BELLOW_CENTER_LEFT_UP:
                return BOTTOM_UPPER_LEFT_CORNER;
            case BELLOW_CENTER_LEFT_DOWN:
                return BOTTOM_LOWER_LEFT_CORNER;
            case BELLOW_CENTER_RIGHT_UP:
                return BOTTOM_UPPER_RIGHT_CORNER;
            case BELLOW_CENTER_RIGHT_DOWN:
                return BOTTOM_LOWER_RIGHT_CORNER;
        }
        throw std::runtime_error("How did we miss a case?");
    }

    struct CuboidStructureCalc {

    public:
        using coordinate_type = Coordinate3D<int>;

        explicit CuboidStructureCalc(const coordinate_type &gridSize);

        explicit CuboidStructureCalc(const CuboidPair &cuboidLocation);


        /// Calculates all faces,edges and corners of a given cuboid
        /// \param cuboid_location Where the cuboid is located
        /// \return Array of pairs that can be indexed using the CuboidFeature enum.
        static std::array<CuboidPair, 38>
        calculate_cuboid_attributes(const CuboidPair &cuboid_location);

        /// Calculates the bottom_left position of all 8 corners
        /// \param cuboid_location Position of the cuboid in space
        /// \return Corners in the following order\n
        /// BOTTOM_LOWER_LEFT_CORNER\n
        /// BOTTOM_LOWER_RIGHT_CORNER\n
        /// BOTTOM_UPPER_LEFT_CORNER\n
        /// BOTTOM_UPPER_RIGHT_CORNER\n
        /// TOP_LOWER_LEFT_CORNER\n
        /// TOP_LOWER_RIGHT_CORNER\n
        /// TOP_UPPER_LEFT_CORNER\n
        /// TOP_UPPER_RIGHT_CORNER\n
        static std::array<Coordinate3D<int>, 8>
        calculate_cuboid_corners(const CuboidPair &cuboid_location);

        const CuboidPair &getCuboidLocation() const;

        const coordinate_type &getCuboidSize() const;

        const std::array<CuboidPair, 38> &getCuboidItems() const;

        /// Extends all features of a cuboid
        /// \param cuboid_attributes a list of all cuboid attributes
        /// \param frame_size size of extension
        /// \return A list containing all objects.
        static std::array<CuboidPair, 38>
        calculate_cuboid_frame_extension(const std::array<CuboidPair, 38> &cuboid_attributes, int frame_size);

        const std::array<CuboidPair, 38> &getCuboidFrameExtension() const;


        /// Determines whether a dimension needs to be sent
        /// \param neighbour_ranks Rank of my neighbours (order defined my neighbour enum iterable)
        /// \param rank rank
        /// \return Ordered list of whether to send or not
        static std::array<bool, 26> should_send(const std::array<std::size_t, 26> &neighbour_ranks, std::size_t rank) {
            std::array<bool, 26> truth_values{};
            for (auto &neighbour: John::NeighbourEnumIterable) {
                truth_values[neighbour] = should_send(neighbour, neighbour_ranks, rank);
            }
            return truth_values;
        }

        constexpr static bool
        should_send(John::NeighbourRelation relation, const std::array<std::size_t, 26> &neighbour_ranks, std::size_t my_rank) {
            return neighbour_ranks[relation] != my_rank;
        }


    protected:
        CuboidPair cuboid_location;
        coordinate_type cuboid_size;
        std::array<CuboidPair, 38> cuboid_items;
        std::array<CuboidPair, 38> cuboid_frame_extension;
        void initialize_self();


    };
} // Geometry

#endif //DISTRIBUTEDBOXREDESIGNED_CUBOIDSTRUCTURECALC_HPP
