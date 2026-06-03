#pragma once

#include <cstdlib>
#include <iostream>

template <typename... Args>
[[noreturn]] void fatalError(Args&&... args)
{
    ((std::cerr << args), ...);
    std::cerr << std::endl;
    std::abort();
}
