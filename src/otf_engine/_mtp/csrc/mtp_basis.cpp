/* -*- c++ -*- ----------------------------------------------------------
   The moment tensor basis of an MTP, from its scalar basis functions.
   Ported from mlip-4 79a39bf, src/pots/mtp/mtp_basis.cpp (BSD 2-Clause).
------------------------------------------------------------------------- */

#include "mtp_basis.h"

#include "mtp_levels.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// A multi-index over {0, 1, 2}^dim, incremented on its last digit first, as mlip-4's.
class Multiindex3 : public std::vector<int> {
  public:
    Multiindex3() = default;
    explicit Multiindex3(int dim) : std::vector<int>(dim), is_valid_(true) {}
    void resize(int dim) { std::vector<int>::resize(dim); is_valid_ = true; }
    Multiindex3& operator++() {
        for (int i = (int) size() - 1; i >= 0; i--)
            if (++(*this)[i] == 3) (*this)[i] = 0;
            else return *this;
        is_valid_ = false;
        return *this;
    }
    int to_ind() const {
        int ret = 0;
        for (int x : *this) ret = 3 * ret + x;
        return ret;
    }
    bool is_valid() const { return is_valid_; }

  private:
    bool is_valid_ = false;
};

int three_pow(int n) {
    int p = 1;
    while (n-- > 0) p *= 3;
    return p;
}

}  // namespace

MTP_Basis_Function::MTP_Basis_Function(std::vector<int> elem) : elem_(std::move(elem)) {
    size_ = (int) std::round(0.5 * (std::sqrt(4.0 * elem_.size() + 1.0) - 1.0));
    if (size_ < 1 || size_ + size_ * size_ != (int) elem_.size())
        throw std::runtime_error("MTP basis function of " + std::to_string(elem_.size()) + " entries: it needs n radial indices and an n x n matrix");
}

bool MTP_Basis_Function::operator<(const MTP_Basis_Function& b) const {
    // compare the vector
    if (size_ < b.size_) return true;
    if (size_ > b.size_) return false;
    for (int i = 0; i < size_; i++) {
        if (elem_[i] < b.elem_[i]) return true;
        if (elem_[i] > b.elem_[i]) return false;
    }

    // compare the matrix
    for (int i = 0; i < size_; i++)
        for (int j = 0; j < size_; j++) {
            if ((*this)(i, j) < b(i, j)) return true;
            if ((*this)(i, j) > b(i, j)) return false;
        }
    return false;
}

bool MTP_Basis_Function::permutation_less(const std::vector<int>& p, const std::vector<int>& q) const {
    for (int i = 0; i < size_; i++) {
        if (elem_[p[i]] < elem_[q[i]]) return true;
        if (elem_[p[i]] > elem_[q[i]]) return false;
    }
    for (int i = 0; i < size_; i++)
        for (int j = 0; j < size_; j++) {
            if ((*this)(p[i], p[j]) < (*this)(q[i], q[j])) return true;
            if ((*this)(p[i], p[j]) > (*this)(q[i], q[j])) return false;
        }
    return false;
}

void MTP_Basis_Function::permute(const std::vector<int>& p) {
    const std::vector<int> swap_elem = elem_;
    for (int i = 0; i < size_; i++)
        elem_[i] = swap_elem[p[i]];
    for (int i = 0; i < size_; i++)
        for (int j = 0; j < size_; j++)
            (*this)(i, j) = swap_elem[size_ + p[i] * size_ + p[j]];
}

std::vector<int> MTP_Basis_Function::Canonize() {
    std::vector<int> perm(size_), best_perm(size_);
    std::iota(perm.begin(), perm.end(), 0);
    std::iota(best_perm.begin(), best_perm.end(), 0);

    for (bool ok = true; ok; ok = std::next_permutation(perm.begin(), perm.end()))
        if (permutation_less(perm, best_perm)) best_perm = perm;
    permute(best_perm);
    return best_perm;
}

int MTP_Basis_Function::Dimension() const {
    int dim = 0;
    for (int i = 0; i < size_; i++) {
        dim += (*this)(i, i);
        for (int j = i + 1; j < size_; j++)
            dim -= 2 * (*this)(i, j);
    }
    return dim;
}

int MTP_Basis_Function_Ext::BestOps(const std::set<MTP_Basis_Function_Ext>& basis_functions) const {
    if (size_ == 1) return 0;
    int best_ops = INT_MAX;
    int best_found = 0;
    const int dim = Dimension();

    subset.assign(size_, 0);
    best_subset.resize(size_);
    subset[size_ - 1] = 1;
    subset_size[0] = size_ - 1;
    subset_size[1] = 1;
    bool ok = true;
    while (ok) {
        int off_diag = 0;
        for (int i = 0; i < size_; i++)
            for (int j = i + 1; j < size_; j++)
                if (subset[i] != subset[j])
                    off_diag += (*this)(i, j);
        const int ops = dim + off_diag;

        // On a tie, the split with more sub-functions already in basis_functions wins
        if (ops <= best_ops) {
            MTP_Basis_Function mtp[2];
            ConstructSubBasisFunctions(mtp);
            const int found_count = (basis_functions.find(mtp[0]) != basis_functions.end()) + (basis_functions.find(mtp[1]) != basis_functions.end());
            if (ops < best_ops || found_count > best_found) {
                best_ops = ops;
                best_subset = subset;
                best_found = found_count;
            }
        }

        // increment a subset
        int i = size_ - 1;
        int zeroth_subset_grew_by = 0;
        for (; subset[i] == 1; i--) {
            subset[i] = 0;
            zeroth_subset_grew_by++;
        }
        subset[i] = 1;
        zeroth_subset_grew_by--;
        subset_size[0] += zeroth_subset_grew_by;
        subset_size[1] -= zeroth_subset_grew_by;
        for (; i >= 0 && subset[i] == 1; i--);
        ok = (i >= 0);
    }
    subset = best_subset;

    subset_size[1] = std::accumulate(subset.begin(), subset.end(), 0);
    subset_size[0] = size_ - subset_size[1];
    return best_ops;
}

void MTP_Basis_Function_Ext::ConstructSubBasisFunctions(MTP_Basis_Function sub_basis_function[2]) const {
    sub_basis_function[0].resize(subset_size[0]);
    sub_basis_function[1].resize(subset_size[1]);

    // Fill the vector of sub_basis_function
    {
        int sub_i[2] = {0, 0};
        for (int i = 0; i < size_; i++)
            sub_basis_function[subset[i]](sub_i[subset[i]]++) = (*this)(i);
    }

    // Fill the matrix of sub_basis_function
    {
        int sub_i[2] = {0, 0};
        for (int i = 0; i < size_; sub_i[subset[i]]++, i++) {
            int sub_j[2] = {0, 0};
            for (int j = 0; j < size_; sub_j[subset[j]]++, j++)
                if (subset[i] == subset[j])
                    sub_basis_function[subset[i]](sub_i[subset[i]], sub_j[subset[j]]) = (*this)(i, j);
        }
    }

    // temp_perm indexes the first sub_basis_function, then the second
    perm.resize(size_);
    std::vector<int> temp_perm(size_);
    {
        int i_perm = 0;
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < size_; i++)
                if (subset[i] == w)
                    temp_perm[i_perm++] = i;
    }

    // Canonize sub_basis_function[w], compose perm according to canonizing permutation
    for (int w = 0; w < 2; w++) {
        const int offset = (w == 0 ? 0 : subset_size[0]);
        const std::vector<int> p = sub_basis_function[w].Canonize();
        for (int i = 0; i < subset_size[w]; i++)
            perm[offset + i] = temp_perm[offset + p[i]];
    }
}

MTP_Basis::MTP_Basis(const std::vector<std::vector<int>>& functions) {
    if (functions.empty())
        throw std::runtime_error("MTP basis: no basis functions");

    // mlip-4's MTP_Basis takes a std::set, so the functions are added in its order
    std::set<MTP_Basis_Function> sorted;
    for (size_t f = 0; f < functions.size(); f++) {
        const MTP_Basis_Function b(functions[f]);
        const std::string which = "MTP basis function " + std::to_string(f);
        for (int i = 0; i < b.size(); i++) {
            if (b(i) < 0)
                throw std::runtime_error(which + ": negative radial index");
            int contracted = 0;
            for (int j = 0; j < b.size(); j++) {
                if (b(i, j) < 0 || b(i, j) != b(j, i))
                    throw std::runtime_error(which + ": the matrix must be symmetric and non-negative");
                if (j != i) contracted += b(i, j);
            }
            if (contracted != b(i, i))
                throw std::runtime_error(which + ": moment " + std::to_string(i) + " has rank " + std::to_string(b(i, i)) + " but shares " + std::to_string(contracted) + " indices; a basis function is a scalar");
        }
        sorted.insert(b);
    }

    for (const auto& b : sorted) AddBasisFunction(b);
    ProcessBasisFunctions();
}

std::set<MTP_Basis_Function_Ext>::iterator MTP_Basis::AddBasisFunction(const MTP_Basis_Function& b) {
    auto where = basis_functions.emplace(b);
    for (int i = 0; i < b.size(); i++)
        radial_function_count_ = std::max(radial_function_count_, 1 + b(i));
    if (!where.second) return where.first;  // check if already exists
    if (where.first->is_elementary()) return where.first;

    // now find out the most efficient way to compute the current function
    // and add the two auxiliary basis functions recursively
    where.first->BestOps(basis_functions);
    MTP_Basis_Function sub[2];
    where.first->ConstructSubBasisFunctions(sub);
    where.first->from[0] = AddBasisFunction(sub[0]);
    where.first->from[1] = AddBasisFunction(sub[1]);
    return where.first;
}

// mlip-4 composes these elements into its program; here each composite element is a sum of
// products of two earlier elements, rows of alpha_index_times.
void MTP_Basis::ProcessBasisFunctions() {
    // Elementary basis functions: the components of one moment tensor, each the moment
    // (mu, px, py, pz) of its index counts
    auto basis_iter = basis_functions.begin();
    for (; basis_iter != basis_functions.end() && basis_iter->size() == 1; ++basis_iter) {
        const auto& b = *basis_iter;
        const int nu = b(0, 0);
        b.element_inds.reserve(three_pow(nu));

        for (Multiindex3 tensor_index(nu); tensor_index.is_valid(); ++tensor_index) {
            std::array<int, 4> curr_moment{{b(0), 0, 0, 0}};
            for (int i = 0; i < nu; i++)
                curr_moment[1 + tensor_index[i]]++;

            const auto where = moment_to_id.emplace(curr_moment, (int) alpha_index_basic.size());
            if (where.second) alpha_index_basic.push_back(curr_moment);
            b.element_inds.push_back(where.first->second);
        }
    }

    // Composite basis functions. A tensor component with the same terms as an earlier one is
    // that element; a scalar basis function keeps an element of its own (see PairMTP::compute).
    int element_count = (int) alpha_index_basic.size();
    std::map<std::map<std::pair<int, int>, int>, int> element_of_terms;
    for (; basis_iter != basis_functions.end(); ++basis_iter) {
        const auto& b = *basis_iter;
        const int dim = b.Dimension();

        const MTP_Basis_Function_Ext& sub_0 = *b.from[0];
        const MTP_Basis_Function_Ext& sub_1 = *b.from[1];
        const int dim_sub[2] = {sub_0.Dimension(), sub_1.Dimension()};
        const int codim = (dim_sub[0] + dim_sub[1] - dim) / 2;  // codimension
        const int tot_dim = dim_sub[0] + dim_sub[1];           // total dimension, before taking contraction

        // effective dimension after the two subbasis functions are computed (but not contracted).
        // Sum of reduced_dim is tot_dim. The i-th moment has reduced_dim[i] indices.
        std::vector<int> reduced_dim(b.size());
        for (int i = 0; i < b.size(); i++)
            reduced_dim[i] = b(i, i);
        for (int i = 0; i < b.size(); i++)
            for (int j = 0; j < b.size(); j++)
                if (i != j && b.subset[i] == b.subset[j])
                    reduced_dim[i] -= b(i, j);

        // ind_map[0] are indices for assignment, ind_map[1] indices for summation
        std::vector<std::vector<int>> ind_map[2];
        ind_map[0].resize(b.size());
        ind_map[1].resize(b.size());
        {
            int ind_count = 0;
            for (int i = 0; i < b.size(); i++) {
                ind_map[0][i].resize(reduced_dim[i]);
                for (int j = 0; j < reduced_dim[i]; j++)
                    ind_map[0][i][j] = ind_count++;
            }

            // Based on the offdiag elements, remove some indices from ind_map[0] and add to ind_map[1]
            const int offset = sub_0.size();
            for (int i = 0; i < sub_0.size(); i++)
                for (int j = 0; j < sub_1.size(); j++)
                    for (int k = 0; k < b(b.perm[i], b.perm[offset + j]); k++) {
                        ind_map[0][b.perm[i]].pop_back();
                        ind_map[0][b.perm[offset + j]].pop_back();
                        ind_map[1][b.perm[i]].push_back(ind_count);
                        ind_map[1][b.perm[offset + j]].push_back(ind_count);
                        ind_count++;
                    }
        }

        // Remap the assignment indices to 0..dim-1; summation indices stay from tot_dim
        int dim_id = 0;
        for (int i = 0; i < b.size(); i++)
            for (int t = 0; t < 2; t++)
                for (int& index : ind_map[t][i])
                    if (index < tot_dim)
                        index = dim_id++;

        // ind_map_set[index_id] will be (w, index in sub_basis_function[w])
        std::vector<std::array<int, 2>> ind_map_set(dim, {{-1, 1}});
        std::vector<int> ind_map_sum[2];
        ind_map_sum[0].assign(codim, -1);
        ind_map_sum[1].assign(codim, -1);
        for (int w = 0; w < 2; w++) {
            const MTP_Basis_Function_Ext& sub = (w == 0 ? sub_0 : sub_1);
            const int offset = (w == 0 ? 0 : sub_0.size());
            int i_sub = 0;
            for (int i = 0; i < sub.size(); i++) {
                const int full_i = b.perm[offset + i];  // position in the full matrix
                for (int index : ind_map[0][full_i])
                    ind_map_set[index] = {{w, i_sub++}};
                for (int index : ind_map[1][full_i])
                    ind_map_sum[w][index - tot_dim] = i_sub++;
            }
        }

        b.element_inds.resize(three_pow(dim));
        Multiindex3 sub_mi[2];
        sub_mi[0].resize(dim_sub[0]);
        sub_mi[1].resize(dim_sub[1]);
        for (Multiindex3 mi(dim); mi.is_valid(); ++mi) {
            for (int a = 0; a < dim; a++)
                sub_mi[ind_map_set[a][0]][ind_map_set[a][1]] = mi[a];

            // mlip-4's MultElement: the count of each product, its two elements in increasing order
            std::map<std::pair<int, int>, int> terms;
            for (Multiindex3 mj(codim); mj.is_valid(); ++mj) {
                for (int c = 0; c < codim; c++) {
                    sub_mi[0][ind_map_sum[0][c]] = mj[c];
                    sub_mi[1][ind_map_sum[1][c]] = mj[c];
                }
                const int left = sub_0.element_inds[sub_mi[0].to_ind()];
                const int right = sub_1.element_inds[sub_mi[1].to_ind()];
                terms[{std::min(left, right), std::max(left, right)}]++;
            }

            if (dim > 0) {
                const auto found = element_of_terms.find(terms);
                if (found != element_of_terms.end()) {
                    b.element_inds[mi.to_ind()] = found->second;
                    continue;
                }
            }
            const int element = element_count++;
            for (const auto& term : terms)
                alpha_index_times.push_back({{term.first.first, term.first.second, term.second, element}});
            if (dim > 0) element_of_terms.emplace(std::move(terms), element);
            b.element_inds[mi.to_ind()] = element;
        }
    }
    alpha_moment_count = element_count;

    // The scalar basis functions, in the set's order: mlip-4's order of the basis coefficients
    for (const auto& b : basis_functions)
        if (b.Dimension() == 0)
            alpha_moment_mapping.push_back(b.element_inds[0]);
}

std::vector<std::vector<int>> level_basis_functions(int level) {
    switch (level) {
        case 6: return MTP6_array;
        case 8: return MTP8_array;
        case 10: return MTP10_array;
        case 12: return MTP12_array;
        case 14: return MTP14_array;
        case 16: return MTP16_array;
        case 18: return MTP18_array;
        case 20: return MTP20_array;
        case 22: return MTP22_array;
        case 24: return MTP24_array;
        case 26: return MTP26_array;
        case 28: return MTP28_array;
        default: throw std::runtime_error("MTP level " + std::to_string(level) + " does not exist: the levels run from 6 to 28 by 2");
    }
}
