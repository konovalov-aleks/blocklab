#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace blocklab {

template <typename Value>
class QuadTree {
private:
    struct Square {
        int32_t x = 0;
        int32_t z = 0;
        int32_t size = 0;
    };

    struct Node;
    using Child = std::unique_ptr<Node>;

    struct Node {
        std::optional<Value> value;
        std::array<Child, 4> children;
    };

public:
    template <typename NodeT, typename ValueT>
    class BasicIterator {
    public:
        BasicIterator() = default;

        ValueT& operator*() const { return *m_current->value; }
        ValueT* operator->() const { return &*m_current->value; }
        int32_t x() const { return m_currentBounds.x; }
        int32_t z() const { return m_currentBounds.z; }

        BasicIterator& operator++()
        {
            advance();
            return *this;
        }

        BasicIterator operator++(int)
        {
            BasicIterator old = *this;
            advance();
            return old;
        }

        friend bool operator==(const BasicIterator& a, const BasicIterator& b) { return a.m_current == b.m_current; }
        friend bool operator!=(const BasicIterator& a, const BasicIterator& b) { return !(a == b); }

    private:
        friend class QuadTree;

        struct Frame {
            NodeT* node = nullptr;
            Square bounds;
            std::size_t nextChild = 0;
        };

        BasicIterator(NodeT* root, const Square& rootBounds)
            : m_root(root)
            , m_rootBounds(rootBounds)
        {
            descend(root, rootBounds);
        }

        BasicIterator(NodeT* root, Square rootBounds, int32_t x, int32_t z)
            : m_root(root)
            , m_rootBounds(rootBounds)
        {
            findLeaf(root, rootBounds, x, z);
        }

        void descend(NodeT* node, Square bounds)
        {
            while (node) {
                if (isLeaf(bounds)) {
                    if (node->value) {
                        m_current = node;
                        m_currentBounds = bounds;
                    }
                    return;
                }

                bool foundChild = false;
                for (std::size_t i = 0; i < node->children.size(); ++i) {
                    auto& child = node->children[i];
                    if (child) {
                        m_stack.push_back({
                            .node = node,
                            .bounds = bounds,
                            .nextChild = i + 1,
                        });
                        node = child.get();
                        bounds = childBounds(bounds, static_cast<int>(i));
                        foundChild = true;
                        break;
                    }
                }

                if (!foundChild)
                    return;
            }
        }

        void findLeaf(NodeT* node, Square bounds, int32_t x, int32_t z)
        {
            while (node) {
                if (isLeaf(bounds)) {
                    if (node->value) {
                        m_current = node;
                        m_currentBounds = bounds;
                    }
                    return;
                }

                const int index = childIndex(bounds, x, z);
                auto& child = node->children[index];
                if (!child)
                    return;

                node = child.get();
                bounds = childBounds(bounds, index);
            }
        }

        void prepareStack()
        {
            if (!m_stack.empty())
                return;

            if (!m_root || !m_current)
                return;

            NodeT* node = m_root;
            Square bounds = m_rootBounds;
            while (node && !isLeaf(bounds)) {
                const int index = childIndex(bounds, m_currentBounds.x, m_currentBounds.z);
                auto& child = node->children[index];
                if (!child) {
                    m_stack.clear();
                    return;
                }

                m_stack.push_back({
                    .node = node,
                    .bounds = bounds,
                    .nextChild = static_cast<std::size_t>(index + 1),
                });
                node = child.get();
                bounds = childBounds(bounds, index);
            }

            if (node != m_current)
                m_stack.clear();
        }

        void advance()
        {
            if (m_current && m_stack.empty())
                prepareStack();

            m_current = nullptr;
            while (!m_stack.empty()) {
                Frame& frame = m_stack.back();
                bool foundChild = false;
                for (std::size_t i = frame.nextChild; i < frame.node->children.size(); ++i) {
                    auto& child = frame.node->children[i];
                    if (child) {
                        frame.nextChild = i + 1;
                        descend(child.get(), childBounds(frame.bounds, static_cast<int>(i)));
                        foundChild = true;
                        break;
                    }
                }

                if (m_current)
                    return;
                if (!foundChild)
                    m_stack.pop_back();
            }
        }

        NodeT* m_current = nullptr;
        Square m_currentBounds;
        NodeT* m_root = nullptr;
        Square m_rootBounds;
        std::vector<Frame> m_stack;
    };

    using iterator = BasicIterator<Node, Value>;
    using const_iterator = BasicIterator<const Node, const Value>;

    struct Rect {
        int32_t x = 0;
        int32_t z = 0;
        int32_t width = 0;
        int32_t height = 0;
    };

    QuadTree() = default;
    QuadTree(const QuadTree& other)
        : m_root(clone(other.m_root))
        , m_rootBounds(other.m_rootBounds)
        , m_size(other.m_size)
    {
    }
    QuadTree(QuadTree&& other) noexcept
        : m_root(std::move(other.m_root))
        , m_rootBounds(std::exchange(other.m_rootBounds, {}))
        , m_size(std::exchange(other.m_size, 0))
    {
    }

    QuadTree& operator=(const QuadTree& other)
    {
        if (this == &other)
            return *this;

        m_root = clone(other.m_root);
        m_rootBounds = other.m_rootBounds;
        m_size = other.m_size;
        return *this;
    }
    QuadTree& operator=(QuadTree&& other) noexcept
    {
        if (this == &other)
            return *this;

        m_root = std::move(other.m_root);
        m_rootBounds = std::exchange(other.m_rootBounds, {});
        m_size = std::exchange(other.m_size, 0);
        return *this;
    }

    bool empty() const { return !m_size; }
    std::size_t size() const { return m_size; }

    void clear()
    {
        m_root.reset();
        m_rootBounds = {};
        m_size = 0;
    }

    void insert(int32_t x, int32_t z, const Value& value) { insertImpl(x, z, Value(value)); }
    void insert(int32_t x, int32_t z, Value&& value) { insertImpl(x, z, std::move(value)); }

    iterator find(int32_t x, int32_t z)
    {
        if (!m_root || !contains(m_rootBounds, x, z))
            return end();
        return iterator(m_root.get(), m_rootBounds, x, z);
    }

    const_iterator find(int32_t x, int32_t z) const
    {
        if (!m_root || !contains(m_rootBounds, x, z))
            return end();
        return const_iterator(static_cast<const Node*>(m_root.get()), m_rootBounds, x, z);
    }

    iterator begin() { return m_root ? iterator(m_root.get(), m_rootBounds) : end(); }
    iterator end() { return iterator(); }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }
    const_iterator cbegin() const { return m_root ? const_iterator(m_root.get(), m_rootBounds) : cend(); }
    const_iterator cend() const { return const_iterator(); }

    iterator erase(iterator position)
    {
        if (position == end())
            return end();

        iterator next = position;
        ++next;
        position.m_current->value.reset();
        --m_size;
        if (!m_size)
            clear();
        return next;
    }

    std::size_t erase(const Rect& rect)
    {
        return eraseIf(rect, [](int32_t, int32_t, const Value&) { return true; });
    }

    template <typename Predicate>
    std::size_t eraseIf(const Rect& rect, Predicate&& predicate)
    {
        if (!m_root || !isValid(rect))
            return 0;

        const std::size_t erased = eraseIfNode(*m_root, m_rootBounds, rect, predicate);
        m_size -= erased;
        if (!m_size)
            clear();
        return erased;
    }

private:
    std::unique_ptr<Node> m_root;
    Square m_rootBounds;
    std::size_t m_size = 0;

    static bool isValid(const Rect& rect) { return rect.width > 0 && rect.height > 0; }
    static bool isLeaf(const Square& bounds) { return bounds.size == 1; }

    static int64_t right(const Rect& rect) { return static_cast<int64_t>(rect.x) + rect.width; }
    static int64_t bottom(const Rect& rect) { return static_cast<int64_t>(rect.z) + rect.height; }
    static int64_t right(const Square& square) { return static_cast<int64_t>(square.x) + square.size; }
    static int64_t bottom(const Square& square) { return static_cast<int64_t>(square.z) + square.size; }

    static bool contains(const Square& square, int32_t x, int32_t z)
    {
        return x >= square.x && z >= square.z && static_cast<int64_t>(x) < right(square)
            && static_cast<int64_t>(z) < bottom(square);
    }

    static bool intersects(const Square& a, const Rect& b)
    {
        return right(a) > b.x && right(b) > a.x && bottom(a) > b.z && bottom(b) > a.z;
    }

    static int childIndex(const Square& parent, int32_t x, int32_t z)
    {
        const int32_t half = parent.size / 2;
        const bool east = x >= parent.x + half;
        const bool south = z >= parent.z + half;
        return (south ? 2 : 0) + (east ? 1 : 0);
    }

    static Square childBounds(const Square& parent, int index)
    {
        const int32_t half = parent.size / 2;
        return {
            .x = parent.x + (index % 2) * half,
            .z = parent.z + (index / 2) * half,
            .size = half,
        };
    }

    template <typename T>
    void insertImpl(int32_t x, int32_t z, T&& value)
    {
        ensureRootContains(x, z);
        if (insertIntoNode(*m_root, m_rootBounds, x, z, std::forward<T>(value)))
            ++m_size;
    }

    void ensureRootContains(int32_t x, int32_t z)
    {
        if (!m_root) {
            m_rootBounds = Square {
                .x = x,
                .z = z,
                .size = 1,
            };
            m_root = std::make_unique<Node>();
            return;
        }

        while (!contains(m_rootBounds, x, z))
            expandRootToward(x, z);
    }

    void expandRootToward(int32_t x, int32_t z)
    {
        const Square oldBounds = m_rootBounds;
        const int32_t oldSize = oldBounds.size;
        const bool expandWest = x < oldBounds.x;
        const bool expandNorth = z < oldBounds.z;
        m_rootBounds = Square {
            .x = oldBounds.x - (expandWest ? oldSize : 0),
            .z = oldBounds.z - (expandNorth ? oldSize : 0),
            .size = oldSize * 2,
        };

        auto newRoot = std::make_unique<Node>();
        const int oldChildIndex = childIndex(m_rootBounds, oldBounds.x, oldBounds.z);
        newRoot->children[oldChildIndex] = std::move(m_root);
        m_root = std::move(newRoot);
    }

    template <typename T>
    static bool insertIntoNode(Node& node, const Square& bounds, int32_t x, int32_t z, T&& value)
    {
        if (isLeaf(bounds)) {
            const bool inserted = !node.value;
            node.value = std::forward<T>(value);
            return inserted;
        }

        const int index = childIndex(bounds, x, z);
        if (!node.children[index])
            node.children[index] = std::make_unique<Node>();
        return insertIntoNode(*node.children[index], childBounds(bounds, index), x, z, std::forward<T>(value));
    }

    template <typename Predicate>
    static std::size_t eraseIfNode(Node& node, const Square& bounds, const Rect& rect, Predicate& predicate)
    {
        if (!intersects(bounds, rect))
            return 0;

        if (isLeaf(bounds)) {
            if (!node.value || !predicate(bounds.x, bounds.z, *node.value))
                return 0;

            node.value.reset();
            return 1;
        }

        std::size_t erased = 0;
        for (std::size_t i = 0; i < node.children.size(); ++i) {
            Child& child = node.children[i];
            if (child)
                erased += eraseIfNode(*child, childBounds(bounds, static_cast<int>(i)), rect, predicate);
        }
        return erased;
    }

    static std::unique_ptr<Node> clone(const std::unique_ptr<Node>& node)
    {
        if (!node)
            return nullptr;

        auto result = std::make_unique<Node>();
        result->value = node->value;
        for (std::size_t i = 0; i < node->children.size(); ++i)
            result->children[i] = clone(node->children[i]);
        return result;
    }
};

} // namespace blocklab
