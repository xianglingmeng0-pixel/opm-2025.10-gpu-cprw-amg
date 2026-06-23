/*
  Copyright 2026 Xiangling Meng.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_GPU_CPRW_WELL_CONTEXT_HEADER_INCLUDED
#define OPM_GPU_CPRW_WELL_CONTEXT_HEADER_INCLUDED

#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/matrixblock.hh>

#include <dune/istl/bcrsmatrix.hh>

#include <memory>

namespace Opm::gpuistl
{

template <class Scalar>
class GpuCprwWellProvider
{
public:
    using PressureMatrix = Dune::BCRSMatrix<MatrixBlock<Scalar, 1, 1>>;

    virtual ~GpuCprwWellProvider() = default;

    virtual int getNumberOfExtraEquations() const = 0;

    virtual void addWellPressureEquationsStruct(PressureMatrix& jacobian) const = 0;

    virtual void addWellPressureEquations(PressureMatrix& jacobian,
                                          const GpuVector<Scalar>& weights,
                                          bool useWellWeights) const = 0;
};

template <class CpuVector>
class GpuCprwCpuWellProvider : public GpuCprwWellProvider<typename CpuVector::field_type>
{
public:
    using Scalar = typename CpuVector::field_type;
    using Base = GpuCprwWellProvider<Scalar>;
    using PressureMatrix = typename Base::PressureMatrix;

    explicit GpuCprwCpuWellProvider(std::unique_ptr<LinearOperatorExtra<CpuVector, CpuVector>> wellOperator)
        : wellOperator_(std::move(wellOperator))
    {
    }

    int getNumberOfExtraEquations() const override
    {
        return wellOperator_->getNumberOfExtraEquations();
    }

    void addWellPressureEquationsStruct(PressureMatrix& jacobian) const override
    {
        wellOperator_->addWellPressureEquationsStruct(jacobian);
    }

    void addWellPressureEquations(PressureMatrix& jacobian,
                                  const GpuVector<Scalar>& weights,
                                  bool useWellWeights) const override
    {
        constexpr auto blockSize = CpuVector::block_type::dimension;
        cpuWeights_.resize(weights.dim() / blockSize);
        weights.copyToHost(cpuWeights_);
        wellOperator_->addWellPressureEquations(jacobian, cpuWeights_, useWellWeights);
    }

private:
    std::unique_ptr<LinearOperatorExtra<CpuVector, CpuVector>> wellOperator_;
    mutable CpuVector cpuWeights_;
};

template <class Scalar>
class GpuCprwWellContext
{
public:
    using Provider = GpuCprwWellProvider<Scalar>;

    static void setProvider(std::shared_ptr<const Provider> provider)
    {
        activeProvider() = std::move(provider);
    }

    static std::shared_ptr<const Provider> getProvider()
    {
        return activeProvider();
    }

private:
    static std::shared_ptr<const Provider>& activeProvider()
    {
        static std::shared_ptr<const Provider> provider;
        return provider;
    }
};

namespace detail
{
    std::shared_ptr<const GpuCprwWellProvider<double>>& activeDoubleCprwWellProvider();
}

template <>
class GpuCprwWellContext<double>
{
public:
    using Provider = GpuCprwWellProvider<double>;

    static void setProvider(std::shared_ptr<const Provider> provider)
    {
        detail::activeDoubleCprwWellProvider() = std::move(provider);
    }

    static std::shared_ptr<const Provider> getProvider()
    {
        return detail::activeDoubleCprwWellProvider();
    }
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_CPRW_WELL_CONTEXT_HEADER_INCLUDED
