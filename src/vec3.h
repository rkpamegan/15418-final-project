/**
 * Original code provided by 15-362 Course Staff, adapted for project use
 * TODO: string/<< operator for both host and device
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <ostream>
#include <assert.h>

#include "vec2.h"
#include "cu_math.h"

// if CPU, "host" and "device" shouldn't mean anything, so define them as nothing
#ifndef __CUDACC__
	#define __host__
	#define __device__
#endif

struct Vec3 { 

	__host__ __device__ Vec3() {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	__host__ __device__ explicit Vec3(float _x, float _y, float _z) {
		x = _x;
		y = _y;
		z = _z;
	}
	__host__ __device__ explicit Vec3(int32_t _x, int32_t _y, int32_t _z) {
		x = static_cast<float>(_x);
		y = static_cast<float>(_y);
		z = static_cast<float>(_z);
	}
	__host__ __device__ explicit Vec3(float f) {
		x = y = z = f;
	}

	Vec3(const Vec3&) = default;
	Vec3& operator=(const Vec3&) = default;
	~Vec3() = default;

	__host__ __device__ float& operator[](uint32_t idx) {
		assert(idx <= 2);
		return data[idx];
	}
	__host__ __device__ float operator[](uint32_t idx) const {
		assert(idx <= 2);
		return data[idx];
	}

	__host__ __device__ Vec3 operator+=(Vec3 v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	__host__ __device__ Vec3 operator-=(Vec3 v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	__host__ __device__ Vec3 operator*=(Vec3 v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}
	__host__ __device__ Vec3 operator/=(Vec3 v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}

	__host__ __device__ Vec3 operator+=(float s) {
		x += s;
		y += s;
		z += s;
		return *this;
	}
	__host__ __device__ Vec3 operator-=(float s) {
		x -= s;
		y -= s;
		z -= s;
		return *this;
	}
	__host__ __device__ Vec3 operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}
	__host__ __device__ Vec3 operator/=(float s) {
		x /= s;
		y /= s;
		z /= s;
		return *this;
	}

	__host__ __device__ Vec3 operator+(Vec3 v) const {
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	__host__ __device__ Vec3 operator-(Vec3 v) const {
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	__host__ __device__ Vec3 operator*(Vec3 v) const {
		return Vec3(x * v.x, y * v.y, z * v.z);
	}
	__host__ __device__ Vec3 operator/(Vec3 v) const {
		return Vec3(x / v.x, y / v.y, z / v.z);
	}

	__host__ __device__ Vec3 operator+(float s) const {
		return Vec3(x + s, y + s, z + s);
	}
	__host__ __device__ Vec3 operator-(float s) const {
		return Vec3(x - s, y - s, z - s);
	}
	__host__ __device__ Vec3 operator*(float s) const {
		return Vec3(x * s, y * s, z * s);
	}
	__host__ __device__ Vec3 operator/(float s) const {
		return Vec3(x / s, y / s, z / s);
	}

	__host__ __device__ bool operator==(Vec3 v) const {
		return x == v.x && y == v.y && z == v.z;
	}
	__host__ __device__ bool operator!=(Vec3 v) const {
		return x != v.x || y != v.y || z != v.z;
	}

	/// Absolute value
	__host__ __device__ Vec3 abs() const {
		return Vec3(cu_abs(x), cu_abs(y), cu_abs(z));
	}
	/// Negation
	__host__ __device__ Vec3 operator-() const {
		return Vec3(-x, -y, -z);
	}
	/// Are all members real numbers?
	__host__ __device__ bool valid() const {
		return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
	}

	/// Modify vec to have unit length
	__host__ __device__ Vec3 normalize() {
		float n = norm();
		x /= n;
		y /= n;
		z /= n;
		return *this;
	}
	/// Return unit length vec in the same direction
	__host__ __device__ Vec3 unit() const {
		float n = norm();
		return Vec3(x / n, y / n, z / n);
	}

	__host__ __device__ float norm_squared() const {
		return x * x + y * y + z * z;
	}
	__host__ __device__ float norm() const {
		return cu_sqrt(norm_squared());
	}
	
	/// Returns first two components
	__host__ __device__ Vec2 xy() const {
		return Vec2(x, y);
	}

	/// Make sure all components are in the range [min,max) with floating point mod logic
	__host__ __device__ Vec3 range(float min, float max) const {
		if (!valid()) return Vec3();
		Vec3 r = *this;
		float range = max - min;
		while (r.x < min) r.x += range;
		while (r.x >= max) r.x -= range;
		while (r.y < min) r.y += range;
		while (r.y >= max) r.y -= range;
		while (r.z < min) r.z += range;
		while (r.z >= max) r.z -= range;
		return r;
	}

	union {
		struct {
			float x;
			float y;
			float z;
		};
		float data[3] = {};
	};
};

__host__ __device__ inline Vec3 operator+(float s, Vec3 v) {
	return Vec3(v.x + s, v.y + s, v.z + s);
}
__host__ __device__ inline Vec3 operator-(float s, Vec3 v) {
	return Vec3(v.x - s, v.y - s, v.z - s);
}
__host__ __device__ inline Vec3 operator*(float s, Vec3 v) {
	return Vec3(v.x * s, v.y * s, v.z * s);
}
__host__ __device__ inline Vec3 operator/(float s, Vec3 v) {
	return Vec3(s / v.x, s / v.y, s / v.z);
}

/// Take minimum of each component
__host__ __device__ inline Vec3 hmin(Vec3 l, Vec3 r) {
	return Vec3(cu_min(l.x, r.x), cu_min(l.y, r.y), cu_min(l.z, r.z));
}

/// Take maximum of each component
__host__ __device__ inline Vec3 hmax(Vec3 l, Vec3 r) {
	return Vec3(cu_max(l.x, r.x), cu_max(l.y, r.y), cu_max(l.z, r.z));
}

/// 3D dot product
__host__ __device__ inline float dot(Vec3 l, Vec3 r) {
	return l.x * r.x + l.y * r.y + l.z * r.z;
}

/// 3D cross product
__host__ __device__ inline Vec3 cross(Vec3 l, Vec3 r) {
	return Vec3(l.y * r.z - l.z * r.y, l.z * r.x - l.x * r.z, l.x * r.y - l.y * r.x);
}

__host__ __device__ inline std::string to_string(Vec3 const &v) {
	return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "}";
}

// __host__ __device__ inline std::ostream& operator<<(std::ostream& out, Vec3 v) {
// 	out << "{" << v.x << "," << v.y << "," << v.z << "}";
// 	return out;
// }

__host__ __device__ inline bool operator<(Vec3 l, Vec3 r) {
	if (l.x == r.x) {
		if (l.y == r.y) {
			return l.z < r.z;
		}
		return l.y < r.y;
	}
	return l.x < r.x;
}
