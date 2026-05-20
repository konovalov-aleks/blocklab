#include "blocklab/characters/PigCharacter.h"

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

    Vec3 pigTarget(Vec3 home, int32_t step)
    {
        static constexpr Vec3 offsets[] {
            { 2.0f, 0.0f, 0.0f },
            { 2.0f, 0.0f, 2.0f },
            { 0.0f, 0.0f, 2.0f },
            { -1.5f, 0.0f, 0.5f },
        };
        return home + offsets[static_cast<std::size_t>(step) % std::size(offsets)];
    }

} // namespace

PigCharacter::PigCharacter(EntityId id, Vec3 position)
    : NPC(id, CharacterKind::Pig, position)
{
    m_radius = 0.35f;
    m_height = 0.8f;
    m_health = 3;
    setAutoJump(true, 7.0f);
    setTurnSpeed(3.5f);
}

void PigCharacter::updateState(World& world, Vec3 threatPosition, float dt)
{
    const float threatDistance = length2D(m_position - threatPosition);
    if (threatDistance < PigThreatDistance && m_state.kind != CharacterStateKind::Panic) {
        setState({
            .kind = CharacterStateKind::Panic,
            .target = { },
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
                .target = pigTarget(m_home, m_walkStep++),
                .timer = 3.0f,
            });
        }
        break;
    case CharacterStateKind::MoveTo:
        moveToward(world, m_state.target, PigWalkSpeed, dt);
        if (m_blockedTimer >= RetargetBlockedTime) {
            setState({
                .kind = CharacterStateKind::Idle,
                .target = { },
                .timer = 0.2f,
            });
            m_blockedTimer = 0.0f;
        } else if (length2D(m_state.target - m_position) <= ArrivalDistance || m_state.timer <= 0.0f) {
            setState({
                .kind = CharacterStateKind::Idle,
                .target = { },
                .timer = 1.25f,
            });
        }
        break;
    case CharacterStateKind::Panic:
        fleeFrom(world, threatPosition, PigPanicSpeed, dt);
        if (m_blockedTimer >= RetargetBlockedTime || m_state.timer <= 0.0f) {
            setState({
                .kind = CharacterStateKind::Idle,
                .target = { },
                .timer = 1.0f,
            });
        }
        break;
    }
}

} // namespace blocklab
