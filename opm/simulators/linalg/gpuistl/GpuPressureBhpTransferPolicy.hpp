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

#ifndef OPM_GPU_PRESSURE_BHP_TRANSFER_POLICY_HEADER_INCLUDED
#define OPM_GPU_PRESSURE_BHP_TRANSFER_POLICY_HEADER_INCLUDED

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/gpuistl/GpuCprwWellContext.hpp>
#include <opm/simulators/linalg/gpuistl/GpuPressureTransferPolicy.hpp>
#include <opm/simulators/linalg/gpuistl/detail/cpr_amg_operations.hpp>
#include <opm/simulators/linalg/twolevelmethodcpr.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Opm::gpuistl
{

template <class FineOperator, class Communication, class Scalar, bool transpose = false>
class GpuPressureBhpTransferPolicy
    : public Dune::Amg::LevelTransferPolicyCpr<FineOperator, Details::GpuCoarseOperatorType<Scalar, Communication>>
{
public:
    using CoarseOperator = typename Details::GpuCoarseOperatorType<Scalar, Communication>;
    using ParentType = Dune::Amg::LevelTransferPolicyCpr<FineOperator, CoarseOperator>;
    using ParallelInformation = Communication;
    using FineVectorType = typename FineOperator::domain_type;
    using CoarseMatrix = typename CoarseOperator::matrix_type;
    using HostPressureMatrix = typename GpuCprwWellProvider<Scalar>::PressureMatrix;
    using WellProvider = GpuCprwWellProvider<Scalar>;

    GpuPressureBhpTransferPolicy(const Communication& comm,
                                 const FineVectorType& weights,
                                 const PropertyTree& prm,
                                 int pressureVarIndex)
        : communication_(&const_cast<Communication&>(comm))
        , weights_(weights)
        , prm_(prm)
        , pressure_var_index_(pressureVarIndex)
    {
    }

    GpuPressureBhpTransferPolicy(const GpuPressureBhpTransferPolicy& other)
        : communication_(other.communication_)
        , weights_(other.weights_)
        , prm_(other.prm_)
        , pressure_var_index_(other.pressure_var_index_)
        , numberOfFineBlocks_(other.numberOfFineBlocks_)
        , wellProvider_(other.wellProvider_)
    {
    }

    void createCoarseLevelSystem(const FineOperator& fineOperator) override
    {
        OPM_TIMEBLOCK(createGpuCprwCoarseLevelSystem);
        const auto& fineLevelMatrix = fineOperator.getmat();
        numberOfFineBlocks_ = fineLevelMatrix.N();
        const bool addWells = prm_.get<bool>("add_wells", false);

        if (addWells) {
            if constexpr (transpose) {
                OPM_THROW(std::logic_error, "GPU CPRWT with BHP coarse rows is not implemented.");
            }
            createPressureBhpCoarseMatrix(fineLevelMatrix);
        } else {
            coarseLevelMatrix_ = std::make_shared<CoarseMatrix>(fineLevelMatrix.getRowIndices(),
                                                                fineLevelMatrix.getColumnIndices(),
                                                                1);
        }

        calculateCoarseEntries(fineOperator);
        coarseLevelCommunication_.reset(communication_, [](Communication*) {});

        this->lhs_.resize(coarseLevelMatrix_->N());
        this->rhs_.resize(coarseLevelMatrix_->N());
        this->operator_ = std::make_shared<CoarseOperator>(*coarseLevelMatrix_);
    }

    void calculateCoarseEntries(const FineOperator& fineOperator) override
    {
        OPM_TIMEBLOCK(calculateGpuCprwCoarseEntries);
        const auto& fineLevelMatrix = fineOperator.getmat();
        const bool addWells = prm_.get<bool>("add_wells", false);

        if (!addWells) {
            coarseLevelMatrix_->getNonZeroValues() = 0.0;
            detail::calculateCoarseEntries<Scalar, transpose>(
                fineLevelMatrix, *coarseLevelMatrix_, weights_, pressure_var_index_);
            return;
        }

        if (!coarseLevelCpuMatrix_ || !fineToCoarseIndex_ || !wellProvider_) {
            OPM_THROW(std::logic_error, "GPU CPRW coarse matrix was not initialized with well rows.");
        }

        clearWellCoarseEntries();
        const bool useWellWeights = prm_.get<bool>("use_well_weights", false);
        wellProvider_->addWellPressureEquations(*coarseLevelCpuMatrix_, weights_, useWellWeights);
        uploadWellCoarseEntries();

        detail::calculateCoarseEntriesMapped<Scalar, transpose>(
            fineLevelMatrix, *coarseLevelMatrix_, weights_, *fineToCoarseIndex_, pressure_var_index_);
        if (wellCoarseEntryValues_ && wellCoarseEntryValues_->dim() > 0) {
            detail::setSparseCoarseEntries(
                *coarseLevelMatrix_, *wellCoarseEntryPositions_, *wellCoarseEntryValues_);
        }
        uploadWellSchurComplementCorrection();
        if (wellSchurComplementValues_ && wellSchurComplementValues_->dim() > 0) {
            detail::addSparseCoarseEntries(
                *coarseLevelMatrix_, *wellSchurComplementPositions_, *wellSchurComplementValues_);
        }
    }

    void moveToCoarseLevel(const typename ParentType::FineRangeType& fine) override
    {
        this->rhs_ = 0;
        detail::restrictVector<Scalar, transpose>(
            fine, this->rhs_, weights_, pressure_var_index_, numberOfFineBlocks_);
        this->lhs_ = 0;
    }

    void moveToFineLevel(typename ParentType::FineDomainType& fine) override
    {
        detail::prolongateVector<Scalar, transpose>(
            this->lhs_, fine, weights_, pressure_var_index_, numberOfFineBlocks_);
    }

    GpuPressureBhpTransferPolicy* clone() const override
    {
        return new GpuPressureBhpTransferPolicy(*this);
    }

    const Communication& getCoarseLevelCommunication() const
    {
        return *coarseLevelCommunication_;
    }

private:
    void createPressureBhpCoarseMatrix(const CoarseMatrix& fineLevelMatrix)
    {
        wellProvider_ = GpuCprwWellContext<Scalar>::getProvider();
        if (!wellProvider_) {
            OPM_THROW(std::logic_error,
                      "GPU CPRW requested add_wells=true, but no GPU CPRW well provider is installed.");
        }

        const auto numberOfWells = wellProvider_->getNumberOfExtraEquations();
        const auto numberOfRows = fineLevelMatrix.N();
        const auto coarseRows = numberOfRows + numberOfWells;
        const auto fineRows = fineLevelMatrix.getRowIndices().asStdVector();
        const auto fineCols = fineLevelMatrix.getColumnIndices().asStdVector();
        const auto averageElementsPerRow = std::max<std::size_t>(1, fineLevelMatrix.nonzeroes() / numberOfRows);

        coarseLevelCpuMatrix_ = std::make_shared<HostPressureMatrix>(coarseRows,
                                                                     coarseRows,
                                                                     averageElementsPerRow,
                                                                     1.2,
                                                                     HostPressureMatrix::implicit);
        for (std::size_t row = 0; row < numberOfRows; ++row) {
            for (int pos = fineRows[row]; pos < fineRows[row + 1]; ++pos) {
                coarseLevelCpuMatrix_->entry(row, fineCols[pos]) = 0.0;
            }
        }

        wellProvider_->addWellPressureEquationsStruct(*coarseLevelCpuMatrix_);
        coarseLevelCpuMatrix_->compress();

        std::vector<int> coarseRowsGpu;
        std::vector<int> coarseColsGpu;
        std::vector<Scalar> coarseValuesGpu;
        std::vector<int> fineToCoarse;
        extractCsrAndFineMap(fineRows, fineCols, coarseRowsGpu, coarseColsGpu, coarseValuesGpu, fineToCoarse);
        coarseRowsHost_ = coarseRowsGpu;
        coarseColsHost_ = coarseColsGpu;
        buildWellCoarseEntryPattern();
        buildSchurComplementPattern();

        coarseLevelMatrix_ = std::make_shared<CoarseMatrix>(coarseValuesGpu.data(),
                                                            coarseRowsGpu.data(),
                                                            coarseColsGpu.data(),
                                                            coarseValuesGpu.size(),
                                                            1,
                                                            coarseRows);
        fineToCoarseIndex_ = std::make_unique<GpuVector<int>>(fineToCoarse);

        if (debugCprw()) {
            std::cout << "[GPU CPRW debug] coarse rows: fine=" << numberOfRows
                      << " wells=" << numberOfWells
                      << " total=" << coarseRows
                      << " nnz=" << coarseValuesGpu.size()
                      << " well_entries=" << wellCoarseEntryPattern_.size()
                      << " schur_entries=" << wellSchurComplementPattern_.size()
                      << '\n';
        }
    }

    int findCoarseEntry(const std::size_t row, const int column) const
    {
        for (int pos = coarseRowsHost_[row]; pos < coarseRowsHost_[row + 1]; ++pos) {
            if (coarseColsHost_[pos] == column) {
                return pos;
            }
        }
        return -1;
    }

    Scalar getCoarseCpuValue(const std::size_t row, const int column) const
    {
        for (auto entry = (*coarseLevelCpuMatrix_)[row].begin();
             entry != (*coarseLevelCpuMatrix_)[row].end();
             ++entry) {
            if (entry.index() == column) {
                return (*entry)[0][0];
            }
        }
        return Scalar(0.0);
    }

    void setCoarseCpuValue(const std::size_t row, const int column, const Scalar value)
    {
        (*coarseLevelCpuMatrix_)[row][column] = value;
    }

    void buildWellCoarseEntryPattern()
    {
        wellCoarseEntryPattern_.clear();
        wellCoarseEntryValuesHost_.clear();
        wellCoarseEntryPositions_.reset();
        wellCoarseEntryValues_.reset();

        const int firstWellColumn = static_cast<int>(numberOfFineBlocks_);
        std::vector<int> positions;

        for (std::size_t row = 0; row + 1 < coarseRowsHost_.size(); ++row) {
            for (int position = coarseRowsHost_[row]; position < coarseRowsHost_[row + 1]; ++position) {
                const int column = coarseColsHost_[position];
                if (row >= numberOfFineBlocks_ || column >= firstWellColumn) {
                    wellCoarseEntryPattern_.push_back({position, row, column});
                    positions.push_back(position);
                }
            }
        }

        if (positions.empty()) {
            return;
        }

        wellCoarseEntryValuesHost_.assign(positions.size(), Scalar(0.0));
        wellCoarseEntryPositions_ = std::make_unique<GpuVector<int>>(positions);
        wellCoarseEntryValues_ = std::make_unique<GpuVector<Scalar>>(wellCoarseEntryValuesHost_);
    }

    void clearWellCoarseEntries()
    {
        for (const auto& entry : wellCoarseEntryPattern_) {
            setCoarseCpuValue(entry.row, entry.column, Scalar(0.0));
        }
    }

    void uploadWellCoarseEntries()
    {
        if (!wellCoarseEntryValues_ || wellCoarseEntryPattern_.empty()) {
            return;
        }

        const int firstWellColumn = static_cast<int>(numberOfFineBlocks_);
        const auto rowScale = bhpRowScale();
        const auto variableScale = bhpVariableScale();

        for (std::size_t index = 0; index < wellCoarseEntryPattern_.size(); ++index) {
            const auto& entry = wellCoarseEntryPattern_[index];
            auto value = getCoarseCpuValue(entry.row, entry.column);
            if (entry.row < numberOfFineBlocks_ && entry.column >= firstWellColumn) {
                value /= variableScale;
            } else if (entry.row >= numberOfFineBlocks_ && entry.column < firstWellColumn) {
                value *= rowScale * variableScale;
            } else if (entry.row >= numberOfFineBlocks_) {
                value *= rowScale;
            }
            wellCoarseEntryValuesHost_[index] = value;
        }

        wellCoarseEntryValues_->copyFromHost(wellCoarseEntryValuesHost_);

        if (shouldPrintDebug(debugWellUploadCount_++)) {
            printSparseStats("well coarse entries",
                             wellCoarseEntryValuesHost_,
                             "bhp_row_scale",
                             rowScale,
                             "bhp_variable_scale",
                             variableScale);
        }
    }

    void buildSchurComplementPattern()
    {
        wellSchurComplementPattern_.clear();
        wellSchurComplementValuesHost_.clear();
        wellSchurComplementPositions_.reset();
        wellSchurComplementValues_.reset();

        const auto numberOfWells = coarseLevelCpuMatrix_->N() - numberOfFineBlocks_;
        if (numberOfWells == 0) {
            return;
        }

        const int firstWellColumn = static_cast<int>(numberOfFineBlocks_);
        std::vector<std::vector<std::size_t>> reservoirRowsByWell(numberOfWells);
        std::vector<std::vector<int>> wellToReservoirColumns(numberOfWells);

        for (std::size_t well = 0; well < numberOfWells; ++well) {
            const auto wellRow = numberOfFineBlocks_ + well;
            for (auto entry = (*coarseLevelCpuMatrix_)[wellRow].begin();
                 entry != (*coarseLevelCpuMatrix_)[wellRow].end();
                 ++entry) {
                const int column = entry.index();
                if (column < firstWellColumn) {
                    wellToReservoirColumns[well].push_back(column);
                }
            }
        }

        for (std::size_t row = 0; row < numberOfFineBlocks_; ++row) {
            for (auto entry = (*coarseLevelCpuMatrix_)[row].begin();
                 entry != (*coarseLevelCpuMatrix_)[row].end();
                 ++entry) {
                const int column = entry.index();
                if (column >= firstWellColumn && column < firstWellColumn + static_cast<int>(numberOfWells)) {
                    reservoirRowsByWell[column - firstWellColumn].push_back(row);
                }
            }
        }

        std::vector<int> positions;
        for (std::size_t well = 0; well < numberOfWells; ++well) {
            for (const auto row : reservoirRowsByWell[well]) {
                for (const auto column : wellToReservoirColumns[well]) {
                    const int position = findCoarseEntry(row, column);
                    if (position >= 0) {
                        wellSchurComplementPattern_.push_back({position, row, well, column});
                        positions.push_back(position);
                    }
                }
            }
        }

        if (positions.empty()) {
            return;
        }

        wellSchurComplementValuesHost_.assign(positions.size(), Scalar(0.0));
        wellSchurComplementPositions_ = std::make_unique<GpuVector<int>>(positions);
        wellSchurComplementValues_ = std::make_unique<GpuVector<Scalar>>(wellSchurComplementValuesHost_);
    }

    void uploadWellSchurComplementCorrection()
    {
        if (!wellSchurComplementValues_ || wellSchurComplementPattern_.empty()) {
            return;
        }

        const auto numberOfWells = coarseLevelCpuMatrix_->N() - numberOfFineBlocks_;
        const int firstWellColumn = static_cast<int>(numberOfFineBlocks_);
        std::vector<Scalar> wellDiagonal(numberOfWells, Scalar(0.0));

        for (std::size_t well = 0; well < numberOfWells; ++well) {
            const auto wellRow = numberOfFineBlocks_ + well;
            const int wellColumn = firstWellColumn + static_cast<int>(well);
            wellDiagonal[well] = getCoarseCpuValue(wellRow, wellColumn);
        }

        for (std::size_t index = 0; index < wellSchurComplementPattern_.size(); ++index) {
            const auto& contribution = wellSchurComplementPattern_[index];
            const int wellColumn = firstWellColumn + static_cast<int>(contribution.well);
            const auto wellRow = numberOfFineBlocks_ + contribution.well;
            const Scalar diagonal = wellDiagonal[contribution.well];

            if (diagonal == Scalar(0.0)) {
                wellSchurComplementValuesHost_[index] = Scalar(0.0);
                continue;
            }

            const Scalar reservoirToWell = getCoarseCpuValue(contribution.reservoirRow, wellColumn);
            const Scalar wellToReservoir = getCoarseCpuValue(wellRow, contribution.reservoirColumn);
            wellSchurComplementValuesHost_[index] = schurCorrectionScale()
                * reservoirToWell * wellToReservoir / diagonal;
        }

        wellSchurComplementValues_->copyFromHost(wellSchurComplementValuesHost_);

        if (shouldPrintDebug(debugSchurUploadCount_++)) {
            printSparseStats("Schur correction",
                             wellSchurComplementValuesHost_,
                             "schur_correction_scale",
                             schurCorrectionScale());
        }
    }

    bool debugCprw() const
    {
        return prm_.get<bool>("debug_cprw", false);
    }

    bool shouldPrintDebug(const std::size_t count) const
    {
        return debugCprw() && (count < 8 || count % 50 == 0);
    }

    Scalar schurCorrectionScale() const
    {
        return static_cast<Scalar>(prm_.get<double>("schur_correction_scale", 1.0));
    }

    Scalar bhpRowScale() const
    {
        return static_cast<Scalar>(prm_.get<double>("bhp_row_scale", 1.0));
    }

    Scalar bhpVariableScale() const
    {
        const auto scale = static_cast<Scalar>(prm_.get<double>("bhp_variable_scale", 1.0));
        if (scale == Scalar(0.0)) {
            OPM_THROW(std::invalid_argument, "GPU CPRW bhp_variable_scale must be nonzero.");
        }
        return scale;
    }

    void printSparseStats(const char* label,
                          const std::vector<Scalar>& values,
                          const char* scaleName,
                          const Scalar scaleValue,
                          const char* secondScaleName = nullptr,
                          const Scalar secondScaleValue = Scalar(0.0)) const
    {
        std::size_t nonzero = 0;
        Scalar absSum = 0.0;
        Scalar signedSum = 0.0;
        Scalar maxAbs = 0.0;

        for (const auto value : values) {
            const auto absValue = static_cast<Scalar>(std::abs(value));
            if (absValue > Scalar(0.0)) {
                ++nonzero;
            }
            absSum += absValue;
            signedSum += value;
            maxAbs = std::max(maxAbs, absValue);
        }

        std::cout << "[GPU CPRW debug] " << label
                  << ": entries=" << values.size()
                  << " nonzero=" << nonzero
                  << " abs_sum=" << absSum
                  << " signed_sum=" << signedSum
                  << " max_abs=" << maxAbs
                  << ' ' << scaleName << '=' << scaleValue
                  ;
        if (secondScaleName) {
            std::cout << ' ' << secondScaleName << '=' << secondScaleValue;
        }
        std::cout << '\n';
    }

    void extractCsrAndFineMap(const std::vector<int>& fineRows,
                              const std::vector<int>& fineCols,
                              std::vector<int>& coarseRows,
                              std::vector<int>& coarseCols,
                              std::vector<Scalar>& coarseValues,
                              std::vector<int>& fineToCoarse) const
    {
        coarseRows.assign(coarseLevelCpuMatrix_->N() + 1, 0);
        coarseCols.clear();
        coarseValues.clear();
        coarseCols.reserve(coarseLevelCpuMatrix_->nonzeroes());
        coarseValues.reserve(coarseLevelCpuMatrix_->nonzeroes());
        fineToCoarse.assign(fineCols.size(), -1);

        int nnz = 0;
        for (auto row = coarseLevelCpuMatrix_->begin(); row != coarseLevelCpuMatrix_->end(); ++row) {
            coarseRows[row.index()] = nnz;
            for (auto entry = row->begin(); entry != row->end(); ++entry) {
                coarseCols.push_back(entry.index());
                coarseValues.push_back((*entry)[0][0]);
                ++nnz;
            }
        }
        coarseRows[coarseLevelCpuMatrix_->N()] = nnz;

        for (std::size_t row = 0; row < numberOfFineBlocks_; ++row) {
            for (int finePos = fineRows[row]; finePos < fineRows[row + 1]; ++finePos) {
                const int column = fineCols[finePos];
                const int coarseBegin = coarseRows[row];
                const int coarseEnd = coarseRows[row + 1];
                const auto coarsePos = std::find(coarseCols.begin() + coarseBegin,
                                                 coarseCols.begin() + coarseEnd,
                                                 column);
                if (coarsePos == coarseCols.begin() + coarseEnd) {
                    OPM_THROW(std::logic_error, "GPU CPRW coarse matrix is missing a reservoir sparsity entry.");
                }
                fineToCoarse[finePos] = std::distance(coarseCols.begin(), coarsePos);
            }
        }
    }

    struct CoarseEntryPattern
    {
        int position;
        std::size_t row;
        int column;
    };

    struct SchurContributionPattern
    {
        int position;
        std::size_t reservoirRow;
        std::size_t well;
        int reservoirColumn;
    };

    Communication* communication_;
    const FineVectorType& weights_;
    PropertyTree prm_;
    const std::size_t pressure_var_index_;
    std::size_t numberOfFineBlocks_ = 0;
    std::shared_ptr<Communication> coarseLevelCommunication_;
    std::shared_ptr<CoarseMatrix> coarseLevelMatrix_;
    std::shared_ptr<HostPressureMatrix> coarseLevelCpuMatrix_;
    std::unique_ptr<GpuVector<int>> fineToCoarseIndex_;
    std::vector<CoarseEntryPattern> wellCoarseEntryPattern_;
    std::unique_ptr<GpuVector<int>> wellCoarseEntryPositions_;
    std::unique_ptr<GpuVector<Scalar>> wellCoarseEntryValues_;
    std::vector<Scalar> wellCoarseEntryValuesHost_;
    std::vector<SchurContributionPattern> wellSchurComplementPattern_;
    std::unique_ptr<GpuVector<int>> wellSchurComplementPositions_;
    std::unique_ptr<GpuVector<Scalar>> wellSchurComplementValues_;
    std::vector<Scalar> wellSchurComplementValuesHost_;
    std::vector<int> coarseRowsHost_;
    std::vector<int> coarseColsHost_;
    std::shared_ptr<const WellProvider> wellProvider_;
    std::size_t debugWellUploadCount_ = 0;
    std::size_t debugSchurUploadCount_ = 0;
};

} // namespace Opm::gpuistl

#endif // OPM_GPU_PRESSURE_BHP_TRANSFER_POLICY_HEADER_INCLUDED
