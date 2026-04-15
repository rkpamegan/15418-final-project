/**
 * Original code provided by 15-362 Course Staff, adapted for project use
 * TODO: string/<< operator for both host and device
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdint.h>
#include <assert.h>

#include "cu_math.h"

#ifndef __CUDACC__
	#define __host__
	#define __device__
#endif

struct Vec2 {

	__host__ __device__ Vec2() {
		x = 0.0f;
		y = 0.0f;
	}
	__host__ __device__ explicit Vec2(float _x, float _y) {
		x = _x;
		y = _y;
	}
	__host__ __device__ explicit Vec2(float f) {
		x = y = f;
	}
	__host__ __device__ explicit Vec2(int32_t _x, int32_t _y) {
		x = static_cast<float>(_x);
		y = static_cast<float>(_y);
	}

	Vec2(const Vec2&) = default;
	Vec2& operator=(const Vec2&) = default;
	~Vec2() = default;

	__host__ __device__ float& operator[](uint32_t idx) {
		assert(idx <= 1);
		return data[idx];
	}
	__host__ __device__ float operator[](uint32_t idx) const {
		assert(idx <= 1);
		return data[idx];
	}

	__host__ __device__ Vec2 operator+=(Vec2 v) {
		x += v.x;
		y += v.y;
		return *this;
	}
	__host__ __device__ Vec2 operator-=(Vec2 v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}
	__host__ __device__ Vec2 operator*=(Vec2 v) {
		x *= v.x;
		y *= v.y;
		return *this;
	}
	__host__ __device__ Vec2 operator/=(Vec2 v) {
		x /= v.x;
		y /= v.y;
		return *this;
	}

	__host__ __device__ Vec2 operator+=(float s) {
		x += s;
		y += s;
		return *this;
	}
	__host__ __device__ Vec2 operator-=(float s) {
		x -= s;
		y -= s;
		return *this;
	}
	__host__ __device__ Vec2 operator*=(float s) {
		x *= s;
		y *= s;
		return *this;
	}
	Vec2 operator/=(float s) {
		x /= s;
		y /= s;
		return *this;
	}

	__host__ __device__ Vec2 operator+(Vec2 v) const {
		return Vec2(x + v.x, y + v.y);
	}
	__host__ __device__ Vec2 operator-(Vec2 v) const {
		return Vec2(x - v.x, y - v.y);
	}
	__host__ __device__ Vec2 operator*(Vec2 v) const {
		return Vec2(x * v.x, y * v.y);
	}
	__host__ __device__ Vec2 operator/(Vec2 v) const {
		return Vec2(x / v.x, y / v.y);
	}

	__host__ __device__ Vec2 operator+(float s) const {
		return Vec2(x + s, y + s);
	}
	__host__ __device__ Vec2 operator-(float s) const {
		return Vec2(x - s, y - s);
	}
	__host__ __device__ Vec2 operator*(float s) const {
		return Vec2(x * s, y * s);
	}
	__host__ __device__ Vec2 operator/(float s) const {
		return Vec2(x / s, y / s);
	}

	__host__ __device__ bool operator==(Vec2 v) const {
		return x == v.x && y == v.y;
	}
	__host__ __device__ bool operator!=(Vec2 v) const {
		return x != v.x || y != v.y;
	}

	/// Absolute value
	__host__ __device__ Vec2 abs() const {
		return Vec2(cu_abs(x), cu_abs(y));
	}
	/// Negation
	__host__ __device__ Vec2 operator-() const {
		return Vec2(-x, -y);
	}
	/// Are all members real numbers?
	__host__ __device__ bool valid() const {
		return std::isfinite(x) && std::isfinite(y);
	}

	/// Modify vec to have unit length
	__host__ __device__ Vec2 normalize() {
		float n = norm();
		x /= n;
		y /= n;
		return *this;
	}
	/// Return unit length vec in the same direction
	__host__ __device__ Vec2 unit() const {
		float n = norm();
		return Vec2(x / n, y / n);
	}

	__host__ __device__ float norm_squared() const {
		return x * x + y * y;
	}
	__host__ __device__ float norm() const {
		return cu_sqrt(norm_squared());
	}

	__host__ __device__ Vec2 range(float min, float max) const {
		if (!valid()) return Vec2();
		Vec2 r = *this;
		float range = max - min;
		while (r.x < min) r.x += range;
		while (r.x >= max) r.x -= range;
		while (r.y < min) r.y += range;
		while (r.y >= max) r.y -= range;
		return r;
	}

	union {
		struct {
			float x;
			float y;
		};
		float data[2] = {};
	};
};

__host__ __device__ inline Vec2 operator+(float s, Vec2 v) {
	return Vec2(v.x + s, v.y + s);
}
__host__ __device__ inline Vec2 operator-(float s, Vec2 v) {
	return Vec2(v.x - s, v.y - s);
}
__host__ __device__ inline Vec2 operator*(float s, Vec2 v) {
	return Vec2(v.x * s, v.y * s);
}
__host__ __device__ inline Vec2 operator/(float s, Vec2 v) {
	return Vec2(s / v.x, s / v.y);
}

/// Take minimum of each component
__host__ __device__ inline Vec2 hmin(Vec2 l, Vec2 r) {
	return Vec2(cu_min(l.x, r.x), cu_min(l.y, r.y));
}
/// Take maximum of each component
__host__ __device__ inline Vec2 hmax(Vec2 l, Vec2 r) {
	return Vec2(cu_max(l.x, r.x), cu_max(l.y, r.y));
}

/// 2D dot product
__host__ __device__ inline float dot(Vec2 l, Vec2 r) {
	return l.x * r.x + l.y * r.y;
}

__host__ __device__ inline std::string to_string(Vec2 const &v) {
	return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
}

__host__ __device__ inline std::ostream& operator<<(std::ostream& out, Vec2 v) {
	out << "{" << v.x << "," << v.y << "}";
	return out;
}
