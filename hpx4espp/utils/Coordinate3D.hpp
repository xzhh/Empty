//
// Created by jn98zk on 6/2/22.
//

#ifndef DISTRIBUTEDBOXREDESIGNED_COORDINATE3D_HPP
#define DISTRIBUTEDBOXREDESIGNED_COORDINATE3D_HPP

#include <array>
#include <iostream>
#include <cmath>
#include "Int3D.hpp"

template<typename T>
struct Coordinate3D {
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmatic type");
    T x{}, y{}, z{};

    /// Converts the coordinate to a raw array
    /// \param arr3d_ptr Raw array ptr with the size of 3
    void to_raw_array(T* arr3d_ptr) const{
        arr3d_ptr[0]  = x;
        arr3d_ptr[1]  = y;
        arr3d_ptr[2]  = z;
    }

    Coordinate3D operator-(const Coordinate3D &cord) const noexcept {
        return Coordinate3D{x - cord.x, y - cord.y, z - cord.z};
    }

    Coordinate3D operator+(const Coordinate3D &cord) const noexcept {
        return Coordinate3D{x + cord.x, y + cord.y, z + cord.z};
    }

    Coordinate3D &operator+=(const Coordinate3D &cord) noexcept {
        x += cord.x;
        y += cord.y;
        z += cord.z;
        return *this;
    }

    Coordinate3D &operator-=(const Coordinate3D &cord) noexcept {
        x -= cord.x;
        y -= cord.y;
        z -= cord.z;
        return *this;
    }

    bool operator==(const Coordinate3D &cord) const noexcept {
        return x == cord.x && y == cord.y && z == cord.z;
    }

    bool operator!=(const Coordinate3D &cord) const noexcept {
        return !(*this == cord);  // NOLINT
    }

    [[nodiscard]] double norm() const noexcept {
        return std::sqrt(x * *x + y * y + z * z);
    }

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & x & y & z;
    }

    friend std::ostream &operator<<(std::ostream &os, const Coordinate3D &c) {
        return os << "Coordinate3D(x=" << c.x << ", y=" << c.y << ", z=" << c.z << ")";
    }

    [[nodiscard]] espressopp::Int3D toInt3D() const{
        return espressopp::Int3D(x,y,z);
    }


    static Coordinate3D<T> fromInt3D(const espressopp::Int3D &input){
        return {input.at(0), input.at(1), input.at(2)};
    }



};

#endif //DISTRIBUTEDBOXREDESIGNED_COORDINATE3D_HPP
