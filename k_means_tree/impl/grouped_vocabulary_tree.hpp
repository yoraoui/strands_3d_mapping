#include "grouped_vocabulary_tree/grouped_vocabulary_tree.h"

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

#include <boost/filesystem.hpp>

using namespace std;

template <typename T>
vector<T> apply_permutation_vector(const vector<T>& vec, const vector<int>& p)
{
    vector<T> sorted_vec(p.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](int i){ return vec[i]; });
    return sorted_vec;
}

template <typename T, typename Compare>
vector<int> sort_permutation_vector(const vector<T>& vec, Compare compare)
{
    vector<int> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](int i, int j){ return compare(vec[i], vec[j]); });
    return p;
}

// we need to add some more argument here that we can request more data from
// actually what we need are cached vocabulary vectors and norms, maybe
// we should store that information with the vocabulary tree, in a folder structure
// this has the advantage that it is independent of what we are representing
// we also need to store the adjacency of subsegments within the sweeps
template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::query_vocabulary(std::vector<result_type>& results, vector<group_type>& groups, CloudPtrT& query_cloud, size_t nbr_query)
{
    // need a way to get
    // 1. mapping - can get this directly but would need to cache
    // 2. vocabulary_norms, vocabulary_vectors and vocabulary_index_vectors, all for one sweep!
    // 3. the clouds - for what do we actually need them? aa, yes, to get the adjacency
    // 4. folder path for saving and loading - part of initialization

    // the result types should probably be structs to make clear which part is which part is which
    std::vector<result_type> scores;
    top_combined_similarities(scores, query_cloud, 50*nbr_query); // make initial number of subsegments configurable

    if (mapping.empty()) {
        super::get_node_mapping(mapping);
    }

    // TODO: fix this part
    // these two must be computed at runtime, probably the first time this function is called
    map<int, node*> inverse_mapping;
    for (const pair<node*, int>& u : mapping) {
        inverse_mapping.insert(make_pair(u.second, u.first));
    }

    std::vector<result_type> updated_scores;
    std::vector<group_type> updated_indices;
    //vector<index_score> total_scores;
    for (size_t i = 0; i < scores.size(); ++i) {
        vector<vocabulary_vector> vectors;
        set<pair<int, int> > adjacencies;

        load_cached_vocabulary_vectors_for_group(vectors, adjacencies, scores[i].group_index);

        vector<int> selected_indices;
        // get<1>(scores[i])) is actually the index within the group!
        double score = super::compute_min_combined_dist(selected_indices, query_cloud, vectors, adjacencies, mapping, inverse_mapping, scores[i].subgroup_index);
        updated_scores.push_back(result_type{ score, scores[i].group_index, scores[i].subgroup_index });
        updated_indices.push_back(selected_indices);
    }

    auto p = sort_permutation_vector(updated_scores, [](const result_type& s1, const result_type& s2) {
        return s1.score < s2.score; // find min elements!
    });

    // the new scores after growing and re-ordering
    results = apply_permutation_vector(updated_scores, p);
    // the subsegment indices within the sweep, not used atm (but we should be able to retrieve this somehow!!)
    groups = apply_permutation_vector(updated_indices, p);

    updated_scores.resize(nbr_query);
    // how should we return the oversegment indices????
    groups.resize(nbr_query);
    for (result_type& s : updated_scores) {
        s.index = get_id_for_group_subgroup(s.group_index, s.subgroup_index);
    }
}


template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::cache_group_adjacencies(int start_ind, vector<set<pair<int, int> > >& adjacencies)
{
    boost::filesystem::path cache_path = boost::filesystem::path(save_state_path) / "vocabulary_vectors";
    boost::filesystem::create_directory(cache_path);

    for (int i = 0; i < adjacencies.size(); ++i) {
        stringstream ss("group");
        ss << setfill('0') << setw(6) << start_ind + i;
        boost::filesystem::path group_path = cache_path / ss.str();
        boost::filesystem::create_directory(group_path);

        boost::filesystem::path adjacencies_path = group_path / "adjacencies.cereal";
        ofstream outa(adjacencies_path.string());
        {
            cereal::BinaryOutputArchive archive_o(outa);
            archive_o(adjacencies[i]);
        }
        outa.close();
    }
}

// takes a vector which for each groups contains the adjacencies of the elements, stores in gvt location
// we need something more here, like at which index the cloud starts
template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::cache_vocabulary_vectors(int start_ind, CloudPtrT& cloud)
{
    // fuck, we need boost to create folders in the adjacencies
    boost::filesystem::path cache_path = boost::filesystem::path(save_state_path) / "vocabulary_vectors";
    boost::filesystem::create_directory(cache_path);

    // we need this to compute the vectors
    if (mapping.empty()) {
        super::get_node_mapping(mapping);
    }

    int first_group = group_subgroup[start_ind].first;
    int current_group = first_group;
    int current_subgroup = 0;
    vector<vocabulary_vector> current_vectors;
    CloudPtrT current_cloud(new CloudT);

    // iterate through all the groups and create the vocabulary vectors and norms
    for (int i = 0; i < cloud->size(); ++i) {
        pair<int, int> group = group_subgroup[start_ind + i]; // this will have to be index in cleaned up cloud
        if (group.first != current_group) {
            save_cached_vocabulary_vectors_for_group(current_vectors, current_group);
            current_group = group.first;
            current_vectors.clear();
        }

        if (group.second != current_subgroup) {
            if (!current_cloud->empty()) {
                vocabulary_vector vec = super::compute_query_index_vector(current_cloud, mapping);
                current_vectors.push_back(vec);
            }
            current_subgroup = group.second;
            current_cloud->clear();
        }
        current_cloud->push_back(cloud->at(i));
        // now, get all of the elements in the group
        // wait 2 s, what happens if we split up one group in the middle?????? we need to make sure that does never happen
        // if group_subgroup was a vector this would be more natural

    }

    // actually, maybe we should do this at the same time as we add the cloud
    // should we demand that the adjacencies are handed in at the same time????
    // yes let's assume that for now
}

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::save_cached_vocabulary_vectors_for_group(vector<vocabulary_vector>& vectors, int i)
{
    boost::filesystem::path cache_path = boost::filesystem::path(save_state_path) / "vocabulary_vectors";

    stringstream ss("group");
    ss << setfill('0') << setw(6) << i;
    boost::filesystem::path group_path = cache_path / ss.str();
    boost::filesystem::create_directory(group_path);

    boost::filesystem::path vectors_path = group_path / "vectors.cereal";
    ofstream outv(vectors_path.string());
    {
        cereal::BinaryOutputArchive archive_o(outv);
        archive_o(vectors);
    }
    outv.close();
}

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::load_cached_vocabulary_vectors_for_group(vector<vocabulary_vector>& vectors,
                                                                                 set<pair<int, int> >& adjacencies, int i)
{
    boost::filesystem::path cache_path = boost::filesystem::path(save_state_path) / "vocabulary_vectors";

    stringstream ss("group");
    ss << setfill('0') << setw(6) << i;
    boost::filesystem::path group_path = cache_path / ss.str();

    boost::filesystem::path vectors_path = group_path / "vectors.cereal";
    ifstream inv(vectors_path.string());
    {
        cereal::BinaryInputArchive archive_i(inv);
        archive_i(vectors);
    }
    inv.close();

    boost::filesystem::path adjacencies_path = group_path / "adjacencies.cereal";
    ifstream ina(adjacencies_path.string());
    {
        cereal::BinaryInputArchive archive_i(ina);
        archive_i(adjacencies);
    }
    ina.close();
}

// the first index is the segment, the second one is the oversegment
template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::set_input_cloud(CloudPtrT& new_cloud, vector<pair<int, int> >& indices)
{
    // to begin with, we might have to remove nan points
    Eigen::Matrix<float, super::rows, 1> p;
    CloudPtrT temp_cloud(new CloudT);
    temp_cloud->reserve(new_cloud->size());
    vector<pair<int, int> > temp_indices;
    temp_indices.reserve(indices.size());

    for (size_t i = 0; i < new_cloud->size(); ++i) {
        p = eig(new_cloud->at(i));
        if (std::find_if(p.data(), p.data()+super::rows, [] (float f) {
            return std::isnan(f) || std::isinf(f);
        }) == p.data()+super::rows) {
            temp_cloud->push_back(new_cloud->at(i));
            temp_indices.push_back(indices[i]);
        }
    }

    cout << "Temp cloud size: " << temp_cloud->size() << endl;
    cout << "Indices size: " << temp_indices.size() << endl;

    //std::sort(temp_indices.begin(), temp_indices.end(), [](const pair<int, int>& p1, const pair<int, int>& p2) {
    //    return p1.first < p2.first && p1.second < p2.second;
    //});

    vector<int> new_indices(temp_indices.size());

    pair<int, int> previous_p = make_pair(-1, -1);
    int index_group_ind = -1;
    int counter = 0;
    for (const pair<int, int>& p : temp_indices) {
        // everything with this index pair should have the same label, assume ordered
        if (p != previous_p) {
            ++index_group_ind;
            group_subgroup[index_group_ind] = p;
            previous_p = p;
        }
        new_indices[counter] = index_group_ind;
        ++counter;
    }

    nbr_points = counter;
    nbr_groups = index_group_ind + 1;

    cout << "New indices size: " << temp_indices.size() << endl;

    super::set_input_cloud(temp_cloud, new_indices);
}

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::append_cloud(CloudPtrT& extra_cloud, vector<pair<int, int> >& indices, bool store_points)
{
    // to begin with, we might have to remove nan points
    Eigen::Matrix<float, super::rows, 1> p;
    CloudPtrT temp_cloud(new CloudT);
    temp_cloud->reserve(extra_cloud->size());
    vector<pair<int, int> > temp_indices;
    temp_indices.reserve(indices.size());

    for (size_t i = 0; i < extra_cloud->size(); ++i) {
        p = eig(extra_cloud->at(i));
        if (std::find_if(p.data(), p.data()+super::rows, [] (float f) {
            return std::isnan(f) || std::isinf(f);
        }) == p.data()+super::rows) {
            temp_cloud->push_back(extra_cloud->at(i));
            temp_indices.push_back(indices[i]);
        }
    }

    // assume these are all new groups
    vector<int> new_indices(temp_indices.size());

    pair<int, int> previous_p = make_pair(-1, -1);
    int index_group_ind = nbr_groups - 1;
    int counter = 0;
    for (const pair<int, int>& p : temp_indices) {
        // everything with this index pair should have the same label, assume ordered
        if (p != previous_p) {
            ++index_group_ind;
            group_subgroup[index_group_ind] = p;
            previous_p = p;
        }
        new_indices[counter] = index_group_ind;
        ++counter;
    }

    nbr_points += counter;
    nbr_groups = index_group_ind + 1;

    super::append_cloud(temp_cloud, new_indices, store_points);
    // here we save our vocabulary vectors in a folder structure
    cache_vocabulary_vectors(nbr_points-temp_cloud->size(), temp_cloud);
}

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::append_cloud(CloudPtrT& extra_cloud, vector<pair<int, int> >& indices, vector<set<pair<int, int> > >& adjacencies, bool store_points)
{
    if (save_state_path.empty()) {
        cout << "If adjacencies are used, need to initialize with the cache path..." << endl;
        exit(-1);
    }
    cache_group_adjacencies(nbr_groups, adjacencies);
    adjacencies.clear();
    append_cloud(extra_cloud, indices, store_points);
}

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::add_points_from_input_cloud(bool save_cloud)
{
    super::add_points_from_input_cloud(true);
    // here we save our vocabulary vectors in a folder structure
    cache_vocabulary_vectors(0, super::cloud);
    if (!save_cloud) {
        super::cloud->clear();
    }
}

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::add_points_from_input_cloud(vector<set<pair<int, int> > >& adjacencies, bool save_cloud)
{
    if (save_state_path.empty()) {
        cout << "If adjacencies are used, need to initialize with the cache path..." << endl;
        exit(-1);
    }
    cache_group_adjacencies(0, adjacencies);
    adjacencies.clear();
    add_points_from_input_cloud(save_cloud);
}

template <typename Point, size_t K>
int grouped_vocabulary_tree<Point, K>::get_id_for_group_subgroup(int group_id, int subgroup_id)
{
    pair<int, int> query(group_id, subgroup_id);
    int ind = -1;
    for (const pair<int, pair<int, int> >& p : group_subgroup) {
        if (p.second == query) {
            ind = p.first;
            break;
        }
    }
    if (ind == -1) {
        cout << "Could not find id corresponding to group/subgroup..." << endl;
        exit(0);
    }
    return ind;
}

template<typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::get_subgroups_for_group(set<int>& subgroups, int group_id)
{
    for (const pair<int, pair<int, int> >& p : group_subgroup) {
        if (p.second.first == group_id) {
            subgroups.insert(p.second.second);
        }
    }
}

/*template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::top_optimized_similarities(vector<tuple<int, int, double> >& scores, CloudPtrT& query_cloud, size_t nbr_results)
{
    std::map<node*, double> query_id_freqs;
    double qnorm = super::compute_query_vector(query_id_freqs, query_cloud);

    unordered_map<int, double> vocabulary_difference(max_index);

    //double epsilon = 7;
    int skipped = 0;
    for (const pair<node*, double>& v : query_id_freqs) {

        unordered_map<int, double> node_vocabulary_vector; // includes weights
        compute_node_vocabulary_vector(node_vocabulary_vector, v.first);

        for (const pair<int, double>& u : node_vocabulary_vector) {
            //present_at_lower_levels.insert(u.first);
            vocabulary_difference[u.first] += std::min(u.second, v.second);
        }
    }

    cout << "Skipped " << float(skipped)/float(query_id_freqs.size()) << " nodes" << endl;

    for (pair<const int, double>& u : vocabulary_difference) {
        //cout << super::db_vector_normalizing_constants[u.first] << endl; // this is always 0
        u.second = 1.0 - u.second/std::max(qnorm, super::db_vector_normalizing_constants[u.first]);
    }

    unordered_map<int, pair<int, double> > map_scores;
    for (const pair<int, double>& s : vocabulary_difference) {
        pair<int, int> grouped_ind = group_subgroup[s.first];
        if (map_scores.count(grouped_ind.first) > 0) {
            pair<int, double>& value = map_scores[grouped_ind.first];
            if (s.second < value.second) {
                value = make_pair(grouped_ind.second, s.second);
            }
        }
        else {
            map_scores.insert(make_pair(grouped_ind.first, make_pair(grouped_ind.second, s.second)));
        }
    }

    // this could probably be optimized a bit also, quite big copy operattion
    //scores.insert(scores.end(), map_scores.begin(), map_scores.end());
    scores.reserve(map_scores.size());
    for (const pair<int, pair<int, double> >& u : map_scores) {
        scores.push_back(make_tuple(u.first, u.second.first, u.second.second));
    }
    std::sort(scores.begin(), scores.end(), [](const tuple<int, int, double>& s1, const tuple<int, int, double>& s2) {
        return get<2>(s1) < get<2>(s2); // find min elements!
    });

    if (nbr_results > 0) {
        scores.resize(nbr_results);
    }
}*/

template <typename Point, size_t K>
void grouped_vocabulary_tree<Point, K>::top_combined_similarities(vector<result_type>& scores, CloudPtrT& query_cloud, size_t nbr_results)
{
    vector<vocabulary_result> smaller_scores;
    super::top_combined_similarities(smaller_scores, query_cloud, 0);
    // this should just be the result_types directly instead
    map<int, pair<int, double> > map_scores;
    for (const vocabulary_result& s : smaller_scores) {
        if (map_scores.size() >= nbr_results) {
            break;
        }
        pair<int, int> groups = group_subgroup[s.index];
        if (map_scores.count(groups.first) > 0) {
            pair<int, double>& value = map_scores[groups.first];
            if (s.score < value.second) {
                value.first = groups.second;
                value.second = s.score;
            }
        }
        else {
            map_scores[groups.first] = make_pair(groups.second, s.score);
        }
    }
    scores.reserve(map_scores.size());
    for (const pair<int, pair<int, double> >& s : map_scores) {
        scores.push_back(result_type{ s.second.second, s.first, s.second.first });
    }
    //scores.insert(scores.end(), map_scores.begin(), map_scores.end());
    std::sort(scores.begin(), scores.end(), [](const result_type& s1, const result_type& s2) {
        return s1.score < s2.score; // find min elements!
    });
    if (nbr_results > 0) {
        scores.resize(nbr_results);
    }
    /*for (result_type& s : scores) {
        s.index = get_id_for_group_subgroup(s.group_index, s.subgroup_index);
    }*/
}

template <typename Point, size_t K>
template <class Archive>
void grouped_vocabulary_tree<Point, K>::load(Archive& archive)
{
    super::load(archive);
    archive(nbr_points, nbr_groups, group_subgroup);
}

template <typename Point, size_t K>
template <class Archive>
void grouped_vocabulary_tree<Point, K>::save(Archive& archive) const
{
    super::save(archive);
    archive(nbr_points, nbr_groups, group_subgroup);
}
