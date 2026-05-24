#include "blocklab/characters/PigCharacter.h"

#include "blocklab/Hash.h"
#include "blocklab/Math.h"
#include "blocklab/World.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace blocklab {

namespace {

    constexpr float PigWalkSpeed = 1.25f;
    constexpr float PigPanicSpeed = 2.75f;
    constexpr float PigThreatDistance = 2.5f;
    constexpr float ArrivalDistance = 0.15f;
    constexpr float RetargetBlockedTime = 0.35f;

    float length2D(Vec3 value) { return std::sqrt(value.x * value.x + value.z * value.z); }

    constexpr uint32_t pigSeed(EntityId id, Vec3 position)
    {
        const uint32_t x = static_cast<uint32_t>(floorToInt(position.x * 16.0f));
        const uint32_t z = static_cast<uint32_t>(floorToInt(position.z * 16.0f));
        return hashCombine(id, x, z);
    }

    Vec3 pigTarget(Vec3 home, uint32_t seed, int32_t step)
    {
        const uint32_t stepSeed = hashCombine(seed, step);
        const float angle = randomFloat01(stepSeed) * Pi * 2.0f;
        const float distance = 1.25f + randomFloat01(stepSeed) * 2.25f;
        return home + Vec3 { std::sin(angle) * distance, 0.0f, std::cos(angle) * distance };
    }

} // namespace

PigCharacter::PigCharacter(EntityId id, Vec3 position)
    : NPC(id, CharacterKind::Pig, position)
    , m_walkSeed(pigSeed(id, position))
{
    m_radius = 0.35f;
    m_height = 0.8f;
    m_health = 3;
    setAutoJump(true, 7.0f);
    setTurnSpeed(3.5f);
    m_state.timer = randomFloat01(hash(m_walkSeed)) * 2.0f;

    const float angle = randomFloat01(m_walkSeed) * Pi * 2.0f;
    m_forward = { std::sin(angle), 0.0f, std::cos(angle) };
}

void PigCharacter::updateState(World& world, Vec3 threatPosition, float dt)
{
    const float threatDistance = length2D(m_position - threatPosition);
    if (threatDistance < PigThreatDistance && m_state.kind != CharacterStateKind::Panic) {
        setState({
            .kind = CharacterStateKind::Panic,
            .target = {},
            .timer = 1.5f,
        });
    }

    m_state.timer = std::max(0.0f, m_state.timer - dt);
    if (horizontalBlocked() && onGround())
        m_blockedTimer += dt;
    else
        m_blockedTimer = 0.0f;

    switch (m_state.kind) {
    case CharacterStateKind::Idle:
    case CharacterStateKind::EatGrass:
    case CharacterStateKind::ChaseTarget:
    case CharacterStateKind::Attack:
        if (m_state.timer <= 0.0f) {
            setState({
                .kind = CharacterStateKind::MoveTo,
                .target = pigTarget(m_home, m_walkSeed, m_walkStep++),
                .timer = 3.0f,
            });
        }
        break;
    case CharacterStateKind::MoveTo:
        moveToward(world, m_state.target, PigWalkSpeed, dt);
        if (m_blockedTimer >= RetargetBlockedTime) {
            setState({
                .kind = CharacterStateKind::Idle,
                .target = {},
                .timer = 0.2f,
            });
            m_blockedTimer = 0.0f;
        } else if (length2D(m_state.target - m_position) <= ArrivalDistance || m_state.timer <= 0.0f) {
            setState({
                .kind = CharacterStateKind::Idle,
                .target = {},
                .timer = 1.25f,
            });
        }
        break;
    case CharacterStateKind::Panic:
        fleeFrom(world, threatPosition, PigPanicSpeed, dt);
        if (m_blockedTimer >= RetargetBlockedTime || m_state.timer <= 0.0f) {
            setState({
                .kind = CharacterStateKind::Idle,
                .target = {},
                .timer = 1.0f,
            });
        }
        break;
    }
}

} // namespace blocklab
