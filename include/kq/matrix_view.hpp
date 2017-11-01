#ifndef MATRIX_VIEW_HPP
#define MATRIX_VIEW_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace kq
{

template<typename T, int NumberOfDimensions>
struct matrix_view;

namespace detail
{

namespace matrix
{
template<typename>
using size_type = long long;

template<size_t NumberOfDimensions>
using coordinates_type = std::array<
    detail::matrix::size_type<void>,
    NumberOfDimensions
>;

template<typename T>
using decayed_tuple_size = std::tuple_size<std::remove_const_t<std::decay_t<T>>>;

template<size_t N, typename T>
constexpr size_type<void> calc_nth_offset_size(T&&);

template<size_t, typename T>
constexpr size_type<void> calc_nth_offset_size_impl(T&&, std::true_type)
{
    return 1;
}

template<size_t N, typename T>
constexpr size_type<void> calc_nth_offset_size_impl(T&& t, std::false_type)
{
    return std::get<N>(t) * calc_nth_offset_size<N-1>(t);
}

template<size_t N, typename T>
constexpr size_type<void> calc_nth_offset_size(T&& t)
{
    constexpr auto S = decayed_tuple_size<T>{};
    using condition = std::integral_constant<bool, S == 0 || N == std::numeric_limits<size_t>::max()>;
    return calc_nth_offset_size_impl<N>(t, condition{});
}

template<typename T, typename U, size_t... Is>
constexpr size_type<void>
calc_final_offset_impl(T&& t, U&& u, std::index_sequence<Is...>)
{
    coordinates_type<sizeof...(Is)+1> vals = {
            u.front(),
            (calc_nth_offset_size<Is>(t) * std::get<Is+1>(u))...
    };
    return std::accumulate(vals.cbegin(), vals.cend(), 0ull, std::plus<>{});
}

template<size_t S>
constexpr size_type<void> calc_final_offset(
        coordinates_type<S> const& sizes,
        coordinates_type<S> const& coordinates)
{
    return calc_final_offset_impl(sizes, coordinates, std::make_index_sequence<S-1>{});
}

struct access
{
    template<typename T>
    static decltype(auto) data(T&& t) {
        return t.data;
    }
    template<typename T>
    static decltype(auto) sizes(T&& t) {
        return t.sizes;
    }
};

template<typename T, int NumberOfDimensions>
struct select_iterable
{
    using type = matrix_view<T, NumberOfDimensions>;
};

template<typename T, int NumberOfDimensions>
using select_iterable_t = typename select_iterable<T, NumberOfDimensions>::type;

template<typename T, int NumberOfDimensions>
struct iterator;

template<typename T, int NumberOfDimensions>
struct select_iterator
{
    using type = iterator<T, NumberOfDimensions>;
};

template<typename T>
struct select_iterator<T, 0>
{
    using type = T*;
};

template<typename T, int NumberOfDimensions>
using select_iterator_t = typename select_iterator<T, NumberOfDimensions>::type;

template<typename T, typename U, size_t... Is>
auto make_iterator_with_indexes(T* ptr, U&& arr, std::index_sequence<Is...>)
{
    return select_iterator_t<T, sizeof...(Is)>{ptr, std::get<Is>(arr)...};
}

template<typename T, int NumberOfDimensions>
struct iterator
{
    using iterable = select_iterable_t<T, NumberOfDimensions>;
    using coord_t = coordinates_type<NumberOfDimensions>;

    template<typename... Us>
    iterator(Us&&... us) : view{std::forward<Us>(us)...} {}

    iterator(iterator const&) = default;
    iterator(iterator&&) = default;
    iterator& operator=(iterator const&) = default;
    iterator& operator=(iterator&&) = default;

    iterator& operator++() {
        return *this += 1;
    }

    iterator& operator--() {
        return *this -= 1;
    }

    iterator operator++(int) {
        auto copy = *this;
        *this += 1;
        return copy;
    }

    iterator operator--(int) {
        auto copy = *this;
        *this -= 1;
        return copy;
    }

    iterator& operator+=(size_type<void> n) {
        constexpr auto S = decayed_tuple_size<coord_t>{};
        return *this = make_iterator_with_indexes(
                    access::data(view) + n * calc_offset(),
                    access::sizes(view),
                    std::make_index_sequence<S>{}
        );
    }

    iterator& operator-=(size_type<void> n) {
        constexpr auto S = decayed_tuple_size<coord_t>{};
        return *this = make_view_with_indexes(
                    access::data(view) - n * calc_offset(),
                    access::sizes(view),
                    std::make_index_sequence<S>{}
        );
    }

    bool operator==(iterator const& o) const {
        return access::data(view) == access::data(o.view);
    }

    bool operator!=(iterator const& o) const {
        return access::data(view) != access::data(o.view);
    }

    bool operator<(iterator const& o) const {
        return access::data(view) < access::data(o.view);
    }

    bool operator<=(iterator const& o) const {
        return access::data(view) <= access::data(o.view);
    }

    bool operator>(iterator const& o) const {
        return access::data(view) > access::data(o.view);
    }

    bool operator>=(iterator const& o) const {
        return access::data(view) >= access::data(o.view);
    }

    iterable const& operator*() const {
        return view;
    }
    iterable& operator*() {
        return view;
    }
    iterable* operator->() const {
        return &view;
    }

private:

    auto calc_offset() const {
        auto const& sizes = access::sizes(view);
        coord_t coordinates{};
        coordinates.back() = view.template dim_size<NumberOfDimensions-1>();
        return calc_final_offset(sizes, coordinates);
    }

    iterable view;
};

template<typename T, int NumberOfDimensions>
auto& operator+(iterator<T, NumberOfDimensions> it, size_type<void> n)
{
    return it += n;
}
template<typename T, int NumberOfDimensions>
auto& operator+(size_type<void> n, iterator<T, NumberOfDimensions> it)
{
    return it += n;
}

template<typename T, int NumberOfDimensions>
auto& operator-(iterator<T, NumberOfDimensions> it, size_type<void> n)
{
    return it -= n;
}
template<typename T, int NumberOfDimensions>
auto& operator-(size_type<void> n, iterator<T, NumberOfDimensions> it)
{
    return it -= n;
}

} // matrix

} // detail

template<typename T, int NumberOfDimensions>
struct matrix_view
{
    static_assert(NumberOfDimensions > 0);

    using coord_t = detail::matrix::coordinates_type<NumberOfDimensions>;
    using size_type = detail::matrix::size_type<void>;

    using iterator = detail::matrix::iterator<T, NumberOfDimensions>;
    using const_iterator = detail::matrix::iterator<T const, NumberOfDimensions>;

    template<typename... Us,
             std::enable_if_t<
                sizeof...(Us) == NumberOfDimensions
             >* = nullptr>
    matrix_view(T* data, Us... sizes):
        matrix_view(data, coord_t{sizes...}) {}

    matrix_view(T* data, coord_t param):
        data{data}, sizes{std::move(param)} {}

    auto begin() const {
        using S = std::make_index_sequence<NumberOfDimensions-1>;
        return detail::matrix::make_iterator_with_indexes(data, sizes, S{});
    }

    auto end() const {
        using S = std::make_index_sequence<NumberOfDimensions-1>;
        coord_t coordinates{};
        coordinates.back() = dim_size<NumberOfDimensions-1>();
        auto offset = detail::matrix::calc_final_offset(sizes, coordinates);
        return detail::matrix::make_iterator_with_indexes(data + offset, sizes, S{});
    }

    template<typename... Us,
             std::enable_if_t<
                sizeof...(Us) == NumberOfDimensions
             >* = nullptr>
    T& operator()(Us... coordinates) {
        return (*this)(coord_t{coordinates...});
    }

    template<typename... Us,
             std::enable_if_t<
                sizeof...(Us) == NumberOfDimensions
             >* = nullptr>
    T const& operator()(Us... coordinates) const {
        return (*this)(coord_t{coordinates...});
    }

    T& operator()(coord_t coordinates) {
        assert_coordinates(coordinates);
        auto offset = detail::matrix::calc_final_offset(sizes, coordinates);
        return data[offset];
    }

    T const& operator()(coord_t coordinates) const {
        assert_coordinates(coordinates);
        auto offset = detail::matrix::calc_final_offset(sizes, coordinates);
        return data[offset];
    }

    template<size_type N>
    size_type dim_size() const {
        return std::get<N>(sizes);
    }

protected:

    void assert_coordinates(coord_t const& coord) const {
#ifdef DEBUG
        if(coord.size() != sizes.size())
            throw std::runtime_error("incorrect coordinates (size mismatch)");
        for(size_t i{}; i < sizes.size(); ++i){
            if(coord[i] >= sizes[i] || coord[i] < 0)
                throw std::runtime_error("incorrect coordinates (out of range)");
        }
#endif
    }

    T* data;
    coord_t sizes;

    friend detail::matrix::access;
};

template<typename T>
struct matrix_2d_view : matrix_view<T, 2>
{
    using matrix_view<T, 2>::matrix_view;

    auto width() const { return this->template dim_size<0>(); }
    auto height() const { return this->template dim_size<1>(); }
};

template<typename T>
struct matrix_3d_view : matrix_view<T, 3>
{
    using matrix_view<T, 3>::matrix_view;

    auto width() const { return this->template dim_size<0>(); }
    auto height() const { return this->template dim_size<1>(); }
    auto depth() const { return this->template dim_size<2>(); }
};

namespace detail
{
namespace matrix
{

template<typename T>
struct select_iterable<T, 2>
{
    using type = matrix_2d_view<T>;
};

template<typename T>
struct select_iterable<T, 3>
{
    using type = matrix_3d_view<T>;
};

} // matrix
} // detail

} // kq

#endif // MATRIX_VIEW_HPP
