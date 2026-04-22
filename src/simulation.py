import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.model_selection
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler
import matplotlib.pyplot as plt
import seaborn as sns
import json

from dataset import CSVDataGenerator
from main import Config, predict

# Define the UserWithPredefBudget class
class UserWithPredefBudget:
    def __init__(
            self,
            config,
            initial_budget,
            default_no_choice_logit=torch.tensor(1.0),
            name="UserWithBudget"
        ):

        self.name = name
        self.slate_size = config.get("slate_size")
        self._num_users = config.get("num_users")

        self.default_no_choice_logit = default_no_choice_logit * torch.ones(self._num_users)
        self.initial_budget = torch.tensor(initial_budget.astype(np.float32))
        self.choice_model = MultinomialLogitChoiceModel(
            batch_shape=(self._num_users,), nochoice_logits=self.default_no_choice_logit)

    def initial_state(self):
        return {'budget': self.initial_budget}

    def generate_watched_vector(self, slate_doc_costs, previous_state):
        cumulative_cost = torch.cumsum(slate_doc_costs, dim=1)
        temp = previous_state['budget'][..., None]
        watched = torch.where(cumulative_cost < temp,
                                torch.ones_like(cumulative_cost),
                                torch.zeros_like(cumulative_cost))
        return watched

    def next_state(self, previous_state, slate_docs):
        return {'budget': previous_state['budget']}

    def next_response(self, previous_state, slate_docs):
        # relevances are in log-scale
        slate_doc_relevances = slate_docs['relevances']
        slate_doc_costs = slate_docs['costs']

        watched = self.generate_watched_vector(slate_doc_costs, previous_state)

        adjusted_scores = watched * slate_doc_relevances
        adjusted_scores_1 = torch.where(adjusted_scores == 0,
                                        torch.tensor(-np.inf) * torch.ones_like(adjusted_scores),
                                        adjusted_scores)

        logits = torch.cat((adjusted_scores_1, self.default_no_choice_logit[..., None]), dim=-1)
        choice = self.choice_model.choice(logits)['choice']
        return {'choice': choice, 'logits': logits, 'watched': watched}

    def observation(self):
        pass

# Define the MultinomialLogitChoiceModel class
class MultinomialLogitChoiceModel:
    def __init__(self, batch_shape, nochoice_logits):
        self.batch_shape = batch_shape
        self.nochoice_logits = nochoice_logits

    def choice(self, logits):
        # logits shape: batch_size x num_choices
        probs = torch.softmax(logits, dim=-1)
        choices = torch.multinomial(probs, 1).squeeze(-1)
        return {'choice': choices}

# Define the RelevanceAndCostCorpus class
class RelevanceAndCostCorpus:
    def __init__(
            self,
            config,
            doc_costs,
            doc_relevances,
            name="RelevanceAndCostCorpus"
        ):
        self.name = name
        self._num_users = config.get("num_users")
        self.doc_costs = torch.tensor(doc_costs.astype(np.float32))
        self.doc_relevances = torch.tensor(doc_relevances.astype(np.float32))

    def initial_state(self):
        return {'relevances': self.doc_relevances, 'costs': self.doc_costs}

    def next_state(self, previous_state):
        return previous_state

    def available_documents(self, corpus_state):
        """The available_documents value."""
        return corpus_state

# Define the PrecomputedSlatesRecommender class
class PrecomputedSlatesRecommender:
    def __init__(
            self,
            config,
            user_slates
        ):
        self.slate_size = config.get("slate_size")
        self._num_users = config.get("num_users")
        self._num_docs = config.get("num_docs")
        self.user_slates = user_slates

    def slate_docs(self, user_state, corpus_state):
        doc_relevances = corpus_state['relevances']
        doc_costs = corpus_state['costs']
        doc_relevances_rep = doc_relevances.unsqueeze(0).repeat(self._num_users, 1)
        doc_costs_rep = doc_costs.unsqueeze(0).repeat(self._num_users, 1)

        indices = torch.tensor(self.user_slates, dtype=torch.int64)
        slate_doc_relevances = torch.gather(doc_relevances_rep, 1, indices)
        slate_doc_costs = torch.gather(doc_costs_rep, 1, indices)

        return {
            'relevances': slate_doc_relevances,
            'costs': slate_doc_costs,
            'indices': indices
        }

def get_simulation_variables(
        num_users=1000,
        horizon=1,
        slate_size=10,
        num_docs=100,
        user_initial_budget=None,
        doc_costs=None,
        doc_relevances=None,
        user_slates=None,
        default_no_choice_logit=4.0,
        seed=0
    ):

    torch.manual_seed(seed)
    num_topics = 2
    config = {
        "slate_size": slate_size,
        "num_users": num_users,
        "num_topics": num_topics,
        "num_docs": num_docs
    }

    # Instantiate the user with predefined budget
    gt_user = UserWithPredefBudget(
        config=config,
        initial_budget=user_initial_budget,
        default_no_choice_logit=torch.tensor(default_no_choice_logit, dtype=torch.float32)
    )


    gt_corpus = RelevanceAndCostCorpus(
        config=config,
        doc_costs=doc_costs,
        doc_relevances=doc_relevances
    )


    gt_recommender = PrecomputedSlatesRecommender(
        config=config,
        user_slates=user_slates
    )

    # Simulate the story
    def my_recs_story(
            config,
            user,
            corpus,
            recommender
        ):
        user_state = user.initial_state()
        corpus_state = corpus.initial_state()
        slate_docs = recommender.slate_docs(user_state, corpus_state)
        user_response = user.next_response(user_state, slate_docs)
        return {
            'user_state': user_state,
            'corpus_state': corpus_state,
            'slate_docs': slate_docs,
            'user_response': user_response
        }

    gt_variables = my_recs_story(config, gt_user, gt_corpus, gt_recommender)
    return gt_variables

def get_simulation_data(
        num_users=1000,
        horizon=1,
        slate_size=10,
        num_docs=100,
        user_initial_budget=None,
        doc_costs=None,
        doc_relevances=None,
        user_slates=None,
        default_no_choice_logit=4.0,
        seed=0
    ):
    gt_variables = get_simulation_variables(
        num_users,
        horizon,
        slate_size,
        num_docs,
        user_initial_budget,
        doc_costs,
        doc_relevances,
        user_slates,
        default_no_choice_logit,
        seed
    )
    traj = {
        'user_state': gt_variables['user_state'],
        'corpus_state': gt_variables['corpus_state'],
        'slate_docs': gt_variables['slate_docs'],
        'user_response': gt_variables['user_response']
    }
    return traj

def compute_metrics(traj, slate_size, num_users):
    choice = traj['user_response']['choice'].numpy()
    slate_level_response = choice != slate_size
    play_rate = slate_level_response.sum() / num_users

    impression = traj['user_response']['watched'].numpy()
    return play_rate, np.mean(impression.sum(axis=1))

def learn_optimal_q_function(
        traj,
        xgb_model,
        call_count,
        discount_factor,
        num_users,
        slate_size,
        all_item_relevances,
        all_item_costs,
        num_iter
    ):

    feature_data = generate_feature_data(num_users, slate_size, traj)

    immediate_reward = feature_data[0]
    stacked_training_features = feature_data[1]
    pregenerate_indices_for_next_state_action = feature_data[2]
    validation_positive_pages_repeated = feature_data[3]
    validation_positive_page_features = feature_data[4]
    validation_negative_pages_repeated = feature_data[5]
    validation_negative_page_features = feature_data[6]
    slates_above_list = feature_data[7]
    # print(slates_above_list)

    xgb_model = train_xgb_model_for_qlearning(
        immediate_reward,
        xgb_model,
        stacked_training_features,
        pregenerate_indices_for_next_state_action,
        call_count,
        discount_factor,
        slates_above_list,
        all_item_relevances,
        all_item_costs,
        num_iter
    )

    auc_score = compute_auc_score(
        xgb_model,
        validation_positive_pages_repeated,
        validation_positive_page_features,
        validation_negative_pages_repeated,
        validation_negative_page_features
    )

    return xgb_model, auc_score

def update_q_function(traj, xgb_model, call_count, discount_factor, num_users, slate_size):

    feature_data = \
        generate_feature_data(num_users, slate_size, traj)

    immediate_reward = feature_data[0]
    stacked_training_features = feature_data[1]
    pregenerate_indices_for_next_state_action = feature_data[2]
    validation_positive_pages_repeated = feature_data[3]
    validation_positive_page_features = feature_data[4]
    validation_negative_pages_repeated = feature_data[5]
    validation_negative_page_features = feature_data[6]

    xgb_model = train_xgb_model(
        immediate_reward,
        xgb_model,
        stacked_training_features,
        pregenerate_indices_for_next_state_action,
        call_count,
        discount_factor
    )

    auc_score = compute_auc_score(
        xgb_model,
        validation_positive_pages_repeated,
        validation_positive_page_features,
        validation_negative_pages_repeated,
        validation_negative_page_features
    )

    return xgb_model, auc_score

def generate_feature_data(num_users, slate_size, traj):

    feature_df = unroll_traj_in_feature_df(num_users, slate_size, traj)

    page_level_labels = feature_df.groupby('page_ind')['label'].sum().reset_index()

    positive_pages = page_level_labels[page_level_labels.label == 1].page_ind.values

    negative_pages = page_level_labels[page_level_labels.label == 0].page_ind.values

    # print('positive_pages.shape, negative_pages.shape: ', positive_pages.shape, negative_pages.shape)

    (training_positive_pages, validation_positive_pages,
     training_negative_pages, validation_negative_pages) = give_train_test_split(positive_pages, negative_pages)

    training_feature_df = \
        feature_df[feature_df.page_ind.isin(np.concatenate([training_positive_pages, training_negative_pages]))]
    training_feature_df = training_feature_df.sort_values(by=['page_ind', 'row_position'])

    immediate_reward = training_feature_df.label.values

    stacked_training_features = featurize(training_feature_df)
    slates_above_list = construct_slates_above(training_feature_df)

    temp = feature_df[feature_df.page_ind.isin(validation_positive_pages)]
    temp = temp.sort_values(by=['page_ind', 'row_position'])

    validation_positive_pages_repeated = temp.page_ind.values

    validation_positive_page_features = featurize(temp)

    temp = feature_df[feature_df.page_ind.isin(validation_negative_pages)]
    temp = temp.sort_values(by=['page_ind', 'row_position'])
    validation_negative_pages_repeated = temp.page_ind.values

    validation_negative_page_features = featurize(temp)

    pre_generated_ids_for_next_state_action = pre_generate_ids_for_next_state_action(
        training_feature_df.page_ind.values,
        np.unique(training_feature_df.page_ind.values)
    )

    return (
        immediate_reward,
        stacked_training_features,
        pre_generated_ids_for_next_state_action,
        validation_positive_pages_repeated,
        validation_positive_page_features,
        validation_negative_pages_repeated,
        validation_negative_page_features,
        slates_above_list
    )

def construct_slates_above(feature_df):
    slates_above_list = []
    slates_above_for_current_page = []
    prev_page_ind = -1
    for ind, row in feature_df.iterrows():
        if row.page_ind != prev_page_ind:
            slates_above_for_current_page = []
        slates_above_list.append(slates_above_for_current_page)
        slates_above_for_current_page = list.copy(slates_above_for_current_page)
        slates_above_for_current_page.append(int(row.item_id))
        prev_page_ind = row.page_ind

    return slates_above_list

def train_xgb_model_for_qlearning(
        immediate_reward,
        xgb_model,
        stacked_training_features,
        pre_generated_ids_for_next_state_action,
        call_count,
        discount_factor,
        slates_above_list,
        all_item_relevances,
        all_item_costs,
        num_iter
    ):

    # repeat each row num_items times
    repeated_stacked_training_features = np.repeat(
        stacked_training_features,
        all_item_relevances.shape[0],
        axis=0
    )
    # repeat the entire relevance and cost array num_samples times
    repeated_relevances = np.tile(all_item_relevances, (stacked_training_features.shape[0]))
    repeated_costs = np.tile(all_item_costs, (stacked_training_features.shape[0]))

    repeated_stacked_training_features[:, 1] = repeated_relevances
    repeated_stacked_training_features[:, 2] = repeated_costs

    # print('call_count', call_count)
    for n_iter in np.arange(num_iter):
        td_targets = compute_qlearning_target(
            immediate_reward,
            xgb_model,
            stacked_training_features.shape[0],
            repeated_stacked_training_features,
            pre_generated_ids_for_next_state_action,
            discount_factor,
            slates_above_list
        )
        if xgb_model is not None:
            # print('warm starting ...')
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1,
                learning_rate=0.1 / call_count,
                n_jobs=24,
                verbosity=0
            ).fit(
                stacked_training_features,
                td_targets,
                xgb_model=xgb_model.get_booster()
            )
        else:
            # print('cold starting ...')
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1,
                n_jobs=24,
                verbosity=0
            ).fit(
                stacked_training_features,
                td_targets
            )

    return xgb_model

def compute_qlearning_target(
        immediate_reward,
        model,
        n_samples,
        repeated_stacked_training_features,
        pregenerate_indices_for_next_state_action,
        discount_factor,
        slates_above_list
    ):
    if model is not None:
        predicted_q_values_all_actions = model.predict(repeated_stacked_training_features)
        predicted_q_values_all_actions = np.reshape(predicted_q_values_all_actions, (n_samples, -1))

        # print(predicted_q_values_all_actions.shape , len(slates_above_list))

        # apply masking
        for ind, slate_above in enumerate(slates_above_list):
            predicted_q_values_all_actions[ind, slate_above] = -np.inf

        optimal_q_values_for_next_step = predicted_q_values_all_actions.max(axis=1)

        target = np.zeros_like(immediate_reward)

        target[pregenerate_indices_for_next_state_action] = \
            optimal_q_values_for_next_step[pregenerate_indices_for_next_state_action + 1]
        target = immediate_reward + discount_factor * target
        return target
    else:
        return np.array(immediate_reward)

def unroll_traj_in_feature_df(num_users, slate_size, traj):

    slate_costs = traj['slate_docs']['costs'].numpy()
    slate_relevance_features = traj['slate_docs']['relevances'].numpy()
    item_ids = traj['slate_docs']['indices'].numpy()
    choice_index = traj['user_response']['choice'].numpy()
    user_budget = traj['user_state']['budget'].numpy()

    label = np.zeros((num_users, slate_size + 1))
    label[np.arange(num_users), choice_index] = 1
    label = label[:, :-1]

    impression = traj['user_response']['watched'].numpy()
    # print('avg impressions,', np.mean(impression.sum(axis=1)))
    x1 = np.repeat(np.arange(num_users), slate_size)
    x2 = np.tile(np.arange(slate_size), num_users)
    x3 = np.ravel(item_ids)

    x4 = np.ravel(slate_relevance_features)
    x5 = np.ravel(impression)
    x6 = np.ravel(label)

    x7 = np.ravel(slate_costs)
    x8 = np.repeat(user_budget, slate_size)

    # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
    feature_df = pd.DataFrame({
        'page_ind': x1,
        'row_position': x2,
        'item_id': x3,
        'relevance': x4,
        'impression': x5,
        'label': x6,
        'cost': x7,
        'user_budget': x8})

    return feature_df

def give_train_test_split(positive_pages, negative_pages):
    smaller_size = min(positive_pages.shape[0], negative_pages.shape[0])

    # training_positive_pages, validation_positive_pages = \
    #     sklearn.model_selection.train_test_split(positive_pages,
    #                                              train_size=int(0.80 * smaller_size),
    #                                              test_size=int(0.2 * smaller_size))
    
    training_positive_pages, validation_positive_pages = \
        sklearn.model_selection.train_test_split(positive_pages,
                                                 test_size=0.2)

    # print('Positive Pages Train/Val Split: ',
    #      training_positive_pages.shape, validation_positive_pages.shape)

    training_negative_pages, validation_negative_pages = \
        sklearn.model_selection.train_test_split(
            negative_pages,
            train_size=training_positive_pages.shape[0],
            test_size=validation_positive_pages.shape[0]
        )

    # print('Negatives Pages Train/Val Split:', training_negative_pages.shape,
    #      validation_negative_pages.shape)

    training_positive_pages = np.sort(training_positive_pages)
    validation_positive_pages = np.sort(validation_positive_pages)

    training_negative_pages = np.sort(training_negative_pages)
    validation_negative_pages = np.sort(validation_negative_pages)

    return (training_positive_pages, validation_positive_pages,
            training_negative_pages, validation_negative_pages)

# row_position, relevance, cost, budget_to_go, cumulative_relevance
def featurize(feature_df):
    return np.hstack(
        [feature_df.row_position.values[..., np.newaxis],
        feature_df.relevance.values[..., np.newaxis],
        feature_df.cost.values[..., np.newaxis],
        get_budget_before_impression(feature_df)[..., np.newaxis],
        get_cumulative_relevance(feature_df)[..., np.newaxis]]
    )

def get_budget_before_impression(feature_df):
    budget_before_impression_all = np.zeros(feature_df.shape[0])
    start_ind = 0
    for p in np.sort(np.unique(feature_df.page_ind.values)):
        costs_for_page_p = \
            feature_df[feature_df.page_ind == p].cost.values

        cumulative_costs = np.cumsum(costs_for_page_p)
        cumulative_costs = np.roll(cumulative_costs, shift=1)
        cumulative_costs[0] = 0.0
        user_budget_for_page_p = feature_df[feature_df.page_ind == p].user_budget.values
        budget_before_impression_all[start_ind: start_ind + user_budget_for_page_p.shape[0]] = \
            user_budget_for_page_p - cumulative_costs

        start_ind += user_budget_for_page_p.shape[0]
    return budget_before_impression_all

def get_cumulative_relevance(feature_df):
    cumulative_relevance = np.zeros(feature_df.shape[0])
    start_ind = 0
    for p in np.sort(np.unique(feature_df.page_ind.values)):
        relevances_for_page_p = \
            feature_df[feature_df.page_ind == p].relevance.values
        relevances_for_page_p = np.roll(relevances_for_page_p, shift=1)
        relevances_for_page_p[0] = 0.0

        cumulative_relevance_for_page_p = np.cumsum(np.exp(relevances_for_page_p))
        cumulative_relevance[start_ind: start_ind + relevances_for_page_p.shape[0]] = \
            cumulative_relevance_for_page_p
        start_ind += relevances_for_page_p.shape[0]
    return cumulative_relevance

def pre_generate_ids_for_next_state_action(all_ids, unique_ids):
    inds_list = []
    for s in unique_ids:
        inds = np.where(all_ids == s)[0]
        if len(inds) > 1:
            inds_list.append(np.arange(inds[0], inds[-1]))
    return np.concatenate(inds_list)

def train_xgb_model(
        immediate_reward,
        xgb_model,
        stacked_training_features,
        pre_generated_ids_for_next_state_action,
        call_count,
        discount_factor
    ):

    # print('call_count', call_count)
    for n_iter in np.arange(1):
        td_targets = compute_td_target_fast(
            immediate_reward,
            xgb_model,
            stacked_training_features,
            pre_generated_ids_for_next_state_action,
            discount_factor
        )
        if xgb_model is not None:
            # print('warm starting ...')
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1,
                learning_rate=0.1 / call_count,
                n_jobs=24,
                verbosity=0
            ).fit(
                stacked_training_features,
                td_targets,
                xgb_model=xgb_model.get_booster()
            )
        else:
            # print('cold starting ...')
            xgb_model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1,
                n_jobs=24,
                verbosity=0
            ).fit(
                stacked_training_features,
                td_targets
            )

    return xgb_model

def compute_td_target_fast(
        immediate_reward,
        model,
        feature_matrix,
        pregenerate_indices_for_next_state_action,
        discount_factor
    ):
    if model is not None:
        target = np.zeros_like(immediate_reward)
        predicted_q_values_next_s_a = model.predict(feature_matrix)
        target[pregenerate_indices_for_next_state_action] = \
            predicted_q_values_next_s_a[pregenerate_indices_for_next_state_action + 1]
        target = immediate_reward + discount_factor * target
        return target
    else:
        return np.array(immediate_reward)

def compute_auc_score(
        xgb_model,
        validation_positive_pages_repeated,
        validation_positive_page_features,
        validation_negative_pages_repeated,
        validation_negative_page_features
    ):

    positive_df = pd.DataFrame({
        'page_ind': validation_positive_pages_repeated,
        'predictions': xgb_model.predict(validation_positive_page_features)
    })

    negative_df = pd.DataFrame({
        'page_ind': validation_negative_pages_repeated,
        'predictions': xgb_model.predict(validation_negative_page_features)
    })

    abandoned_pages_score = negative_df.groupby('page_ind')['predictions'].apply(np.mean).reset_index()
    novel_play_page_score = positive_df.groupby('page_ind')['predictions'].apply(np.mean).reset_index()

    y_true = np.concatenate([np.ones_like(novel_play_page_score['predictions'].values),
                             np.zeros_like(abandoned_pages_score['predictions'].values)])

    y_pred = np.concatenate([novel_play_page_score['predictions'].values,
                             abandoned_pages_score['predictions'].values])

    return roc_auc_score(y_true=y_true, y_score=y_pred)

def generate_user_slates(
        xgb_model,
        all_user_budgets,
        relevance_all_items,
        epsilon,
        cost_all_items,
        num_users,
        slate_size,
        num_docs,
        seed
    ):
    block_level_states = \
        Parallel(n_jobs=32)(delayed(generate_user_slate_block)(
            xgb_model,
            all_user_budgets,
            relevance_all_items,
            epsilon,
            cost_all_items,
            num_users,
            slate_size,
            num_docs,
            seed,
            block_ind
        ) for block_ind in np.arange(32))

    user_slates_a = np.zeros((num_users, slate_size), np.int32)
    for block in block_level_states:
        for u in block:
            user_slates_a[u[0], :] = np.array(u[1])
    return user_slates_a

def generate_user_slate_block(
        xgb_model,
        all_user_budgets,
        relevance_all_items,
        epsilon,
        cost_all_items,
        num_users,
        slate_size,
        num_docs,
        seed,
        block_ind
    ):
    part_size = int(np.ceil(num_users / 32))
    max_ind = min((block_ind + 1) * part_size, num_users)
    block_level_states = []
    for nu in np.arange(block_ind * part_size, max_ind):
        block_level_states.append(generate_user_slate(
            xgb_model, all_user_budgets,
            relevance_all_items, epsilon,
            cost_all_items, slate_size, num_docs, nu, seed
        ))
    return block_level_states

def generate_user_slate(
        xgb_model,
        all_user_budgets,
        relevance_all_items,
        epsilon,
        cost_all_items,
        slate_size,
        num_docs,
        user_ind,
        seed
    ):
    chosen_actions = []
    user_budget_to_go = all_user_budgets[user_ind]
    cumulative_relevance_so_far = np.exp(0)
    for k in np.arange(slate_size):
        feature_matrix = generate_feature_matrix(
            k,
            relevance_all_items,
            cost_all_items,
            user_budget_to_go,
            cumulative_relevance_so_far,
            num_docs
        )
        predicted_q_values = xgb_model.predict(feature_matrix)
        predicted_q_values[chosen_actions] = -np.inf
        chosen_action = epsilon_greedy_policy(predicted_q_values, epsilon, chosen_actions, seed)
        chosen_actions.append(chosen_action)
        user_budget_to_go -= cost_all_items[chosen_action]
        cumulative_relevance_so_far += np.exp(relevance_all_items[chosen_action])
    return user_ind, chosen_actions

# row_position, relevance, cost, budget_to_go
def generate_feature_matrix(
        row_position,
        relevance_all_items,
        cost_all_items,
        budget_to_go,
        cumulative_relevance_so_far,
        num_docs
    ):

    return np.hstack(
        [np.repeat(row_position, num_docs)[..., np.newaxis],
        relevance_all_items[..., np.newaxis],
        cost_all_items[..., np.newaxis],
        np.repeat(budget_to_go, num_docs)[..., np.newaxis],
        np.repeat(cumulative_relevance_so_far, num_docs)[..., np.newaxis]]
    )

def epsilon_greedy_policy(predicted_q_values, epsilon, current_slate, seed):
    probs = get_epsilon_greedy_policy_prob(predicted_q_values, current_slate, epsilon)
    np.random.seed(seed)
    return np.random.choice(predicted_q_values.shape[0], size=1, replace=True, p=probs)[0]

def get_epsilon_greedy_policy_prob(predicted_q_values, current_slate, epsilon):
    best_action = np.argmax(predicted_q_values)

    remaining_slate_size = predicted_q_values.shape[0] - len(current_slate)
    probs = np.ones(predicted_q_values.shape[0]) * (epsilon / remaining_slate_size)
    probs[current_slate] = 0
    probs[best_action] += (1 - epsilon)
    return probs

def get_random_user_slates(num_users, slate_size, num_docs):
    user_slates = np.zeros((num_users, slate_size), np.int32)
    for nu in np.arange(num_users):
        user_slates[nu, :] = np.random.choice(np.arange(num_docs), size=slate_size, replace=False)

    return user_slates

def run_sarsa(
        num_users,
        slate_size,
        num_docs,
        user_initial_budget,
        doc_costs,
        doc_relevances,
        discount_factor,
        num_iter=25,
        seed=0
    ):
    np.random.seed(seed)
    xgb_model = None
    epsilon = 0.1
    user_slates = get_random_user_slates(num_users, slate_size, num_docs)
    # print(user_slates[0])
    results = []
    traj_list = []
    for iter_ind in range(num_iter):
        print(f"Iteration {iter_ind + 1}/{num_iter}: Generating simulated data ...")

        traj = get_simulation_data(
            num_users=num_users,
            slate_size=slate_size,
            num_docs=num_docs,
            user_initial_budget=user_initial_budget,
            doc_costs=doc_costs,
            doc_relevances=doc_relevances,
            user_slates=user_slates,
            default_no_choice_logit=4.0,
            seed=seed
        )

        metrics = compute_metrics(traj, slate_size, num_users)
        xgb_model, auc_score = update_q_function(
            traj=traj,
            xgb_model=xgb_model,
            call_count=iter_ind + 1,
            discount_factor=discount_factor,
            num_users=num_users,
            slate_size=slate_size
        )

        print(f"Iteration {iter_ind + 1}/{num_iter}: Play Rate: {metrics[0]}, Avg Impressions: {metrics[1]}, AUC: {auc_score}")
        print(f"Iteration {iter_ind + 1}/{num_iter}: Generating user slates ...")

        user_slates = generate_user_slates(
            xgb_model=xgb_model,
            all_user_budgets=user_initial_budget,
            relevance_all_items=doc_relevances,
            epsilon=epsilon,
            cost_all_items=doc_costs,
            num_users=num_users,
            slate_size=slate_size,
            num_docs=num_docs,
            seed=seed
        )

        # print(user_slates[0])
        results.append(metrics)
        # traj_list.append(traj)

    return results, xgb_model, traj_list

def run_qlearning(
        num_users,
        slate_size,
        num_docs,
        user_initial_budget,
        doc_costs,
        doc_relevances,
        discount_factor,
        num_iter=25,
        seed=0,
        print_logs=False
    ):

    np.random.seed(seed)
    xgb_model = None
    epsilon = 0.1
    user_slates_from_behavioral_policy = get_random_user_slates(num_users, slate_size, num_docs)
    results = []
    traj_list = []

    if (print_logs):
        print(f'Generating Simulated Data ...')

    behavior_policy_traj = get_simulation_data(
        num_users=num_users,
        slate_size=slate_size,
        num_docs=num_docs,
        user_initial_budget=user_initial_budget,
        doc_costs=doc_costs,
        doc_relevances=doc_relevances,
        user_slates=user_slates_from_behavioral_policy,
        default_no_choice_logit=4.0,
        seed=seed
    )

    behavior_policy_metrics = compute_metrics(behavior_policy_traj, slate_size, num_users)

    if (print_logs):
        print(f"Behavior Policy Metrics - Play Rate: {behavior_policy_metrics[0]}, Avg Impressions: {behavior_policy_metrics[1]}")

    xgb_model, auc_score = learn_optimal_q_function(
        behavior_policy_traj,
        xgb_model,
        1,
        discount_factor,
        num_users,
        slate_size,
        doc_relevances,
        doc_costs,
        num_iter
    )

    if (print_logs):
        print('Generating User Slates from QLearning Policy ...')

    user_slates_from_qlearning_policy = generate_user_slates(
        xgb_model=xgb_model,
        all_user_budgets=user_initial_budget,
        relevance_all_items=doc_relevances,
        epsilon=epsilon,
        cost_all_items=doc_costs,
        num_users=num_users,
        slate_size=slate_size,
        num_docs=num_docs,
        seed=seed
    )

    qlearning_policy_traj = get_simulation_data(
        num_users=num_users,
        slate_size=slate_size,
        num_docs=num_docs,
        user_initial_budget=user_initial_budget,
        doc_costs=doc_costs,
        doc_relevances=doc_relevances,
        user_slates=user_slates_from_qlearning_policy,
        default_no_choice_logit=4.0,
        seed=seed
    )

    qlearning_policy_metrics = compute_metrics(qlearning_policy_traj, slate_size, num_users)

    if (print_logs):
        print(f"Play Rate: {qlearning_policy_metrics[0]}, Avg Impressions: {qlearning_policy_metrics[1]}, AUC: {auc_score}")

    # print(user_slates[0])
    results.append(qlearning_policy_metrics)
    traj_list.append(qlearning_policy_traj)

    return results

class Params(object):
    def __init__(
            self,
            budget,
            discount_factor,
            num_users,
            num_docs,
            slate_size,
            epsilon,
            seed
        ):
        self.budget = budget
        self.discount_factor = discount_factor
        self.num_users = num_users
        self.num_docs = num_docs
        self.slate_size = slate_size
        self.epsilon = epsilon
        self.seed = seed

    def __str__(self):
        return f'''budget : {self.budget}, discount_factor : {self.discount_factor},
        num_users : {self.num_users}, num_docs : {self.num_docs},
        slate_size : {self.slate_size}, epsilon : {self.epsilon}, seed : {self.seed}'''

    def __hash__(self):
        return hash(
            (self.budget, self.discount_factor,
            self.num_users, self.num_docs,
            self.slate_size, self.epsilon,
            self.seed)
        )

    def __eq__(self, other):
        return ((self.budget, self.discount_factor, self.num_users,
                 self.num_docs, self.slate_size, self.epsilon, self.seed)) == \
               ((other.budget, other.discount_factor, other.num_users, other.num_docs,
                 other.slate_size, other.epsilon, other.seed))
    
def run_sarsa_for_seed(budget, discount_factor, seed, num_iterations_for_model_fitting, num_users, item_relevances):
    print(f"====== SARSA Experiement Start - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    np.random.seed(seed)
    behavior_policy_params = Params(
        num_users=num_users,
        num_docs=num_items,
        slate_size=slate_size,
        epsilon=epsilon,
        budget=budget,
        seed=seed,
        discount_factor=discount_factor
    )
    all_user_budgets = np.random.lognormal(
        np.log(behavior_policy_params.budget),
        user_budget_scale,
        behavior_policy_params.num_users
    )

    relevance_all_items = item_relevances

    cost_all_items = np.random.uniform(
        cost_low,
        cost_high,
        size=behavior_policy_params.num_docs
    )

    experiment_results = run_sarsa(
        num_users=behavior_policy_params.num_users,
        slate_size=behavior_policy_params.slate_size,
        num_docs=behavior_policy_params.num_docs,
        user_initial_budget=all_user_budgets,
        doc_costs=cost_all_items,
        doc_relevances=relevance_all_items,
        discount_factor=behavior_policy_params.discount_factor,
        num_iter=num_iterations_for_model_fitting,
        seed=behavior_policy_params.seed
    )

    print(f"====== SARSA Experiement End - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    return behavior_policy_params, experiment_results[0]

def run_qlearning_for_seed(budget, discount_factor, seed, num_iter, num_users, item_relevances):
    print(f"====== Q-Learning Experiement Start - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    np.random.seed(seed)
    behavior_policy_params = Params(
        num_users=num_users,
        num_docs=num_items,
        slate_size=slate_size,
        epsilon=epsilon,
        budget=budget,
        seed=seed,
        discount_factor=discount_factor
    )
    all_user_budgets = np.random.lognormal(
        np.log(behavior_policy_params.budget),
        user_budget_scale,
        behavior_policy_params.num_users
    )
    
    relevance_all_items = item_relevances

    cost_all_items = np.random.uniform(
        cost_low,
        cost_high,
        size=behavior_policy_params.num_docs
    )

    results = run_qlearning(
        num_users=behavior_policy_params.num_users,
        slate_size=behavior_policy_params.slate_size,
        num_docs=behavior_policy_params.num_docs,
        user_initial_budget=all_user_budgets,
        doc_costs=cost_all_items,
        doc_relevances=relevance_all_items,
        discount_factor=behavior_policy_params.discount_factor,
        num_iter=num_iter,
        seed=behavior_policy_params.seed,
        print_logs=True
    )

    print(f"====== Q-Learning Experiement End - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")

    return behavior_policy_params, results

def json_to_dataframe(file_path):
    """
    Reads a JSON file containing a dictionary of results, parses it, and converts it into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the 'results_sarsa.json' or 'results_qlearning.json' file.

    Returns:
    - pd.DataFrame: A DataFrame with extracted parameters and corresponding values.
    """
    # Read the content of the JSON file
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        # Safely evaluate the string content to a Python dictionary
        data_dict = ast.literal_eval(content)
    except Exception as e:
        raise ValueError("Error parsing the JSON file. Ensure it contains a valid dictionary.") from e
    
    records = []
    
    for key, value in data_dict.items():
        # Initialize a dictionary to hold parameters for each record
        params = {}
        
        # Split the key string into individual parameter strings
        parts = key.split(',')
        for part in parts:
            # Remove any newline characters and extra spaces
            part = part.strip().replace('\n', '')
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip()
                v = v.strip()
                params[k] = v
        
        # Convert numeric parameters to appropriate types
        for param in ['budget', 'discount_factor', 'num_users', 'num_docs', 'slate_size', 'epsilon', 'seed']:
            if param in params:
                try:
                    # Convert to float if there's a decimal point, else to int
                    if '.' in params[param]:
                        params[param] = float(params[param])
                    else:
                        params[param] = int(params[param])
                except ValueError:
                    # If conversion fails, keep the original string
                    pass
        
        # Extract 'play_rate' and 'effective_slate_size' by averaging if multiple entries exist
        if isinstance(value, list) and len(value) >= 2:
            final_play_rate = value[-1][0]
            final_imp_rate = value[-1][1]
            play_rate, effective_slate_size = final_play_rate, final_imp_rate
        elif isinstance(value, list) and len(value) == 1:
            play_rate, effective_slate_size = value[0]
        else:
            play_rate = None
            effective_slate_size = None
        
        params['play_rate'] = play_rate
        params['effective_slate_size'] = effective_slate_size
        
        # Append the parameters dictionary to the records list
        records.append(params)
    
    # Create a DataFrame from the list of records
    df = pd.DataFrame(records)
    
    # Rename 'budget' to 'user_budget' for clarity
    if 'budget' in df.columns:
        df = df.rename(columns={'budget': 'user_budget'})
    
    return df

def _load_all_results():
    """Load all five algorithm result files into a single combined DataFrame."""
    algo_files = {
        'SARSA':      '../outputs/results_sarsa.json',
        'Q-Learning': '../outputs/results_qlearning.json',
        'PPO':        '../outputs/results_ppo.json',
        'GRPO':       '../outputs/results_grpo.json',
        'GSPO':       '../outputs/results_gspo.json',
    }
    dfs = []
    for algo, path in algo_files.items():
        try:
            df = json_to_dataframe(path)
            df['algorithm'] = algo
            dfs.append(df)
        except FileNotFoundError:
            print(f"[warning] {path} not found – skipping {algo}")
    return pd.concat(dfs, ignore_index=True)


def plot_results():
    """Plot Play Rate and Effective Slate Size vs Discount Factor for all algorithms."""
    sns.set_theme(style="whitegrid")
    combined_df = _load_all_results()
    print(combined_df.head())

    user_budgets = np.sort(combined_df['user_budget'].unique())
    print("User Budgets:", user_budgets)

    algo_order = ['SARSA', 'Q-Learning', 'PPO', 'GRPO', 'GSPO']
    palette = sns.color_palette("tab10", n_colors=len(algo_order))

    # --- Play Rate vs Discount Factor ---
    f, axes = plt.subplots(nrows=1, ncols=len(user_budgets),
                           figsize=(4 * len(user_budgets), 5),
                           sharex=True, sharey=True, squeeze=False)
    for n, ub in enumerate(user_budgets):
        ax = axes[0][n]
        sub = combined_df[combined_df.user_budget == ub]
        sns.lineplot(ax=ax, data=sub, x='discount_factor', y='play_rate',
                     hue='algorithm', hue_order=algo_order, palette=palette,
                     marker='o', errorbar='se')
        ax.set_title(f'Budget = {ub}')
        ax.set_xlabel('Discount Factor')
        ax.set_ylabel('Play Rate' if n == 0 else '')
        if n > 0:
            ax.get_legend().remove()
    axes[0][-1].legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.suptitle('Play Rate vs Discount Factor', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("../outputs/playrate_v_budget_all.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Effective Slate Size vs Discount Factor ---
    f, axes = plt.subplots(nrows=1, ncols=len(user_budgets),
                           figsize=(4 * len(user_budgets), 5),
                           sharex=True, sharey=True, squeeze=False)
    for n, ub in enumerate(user_budgets):
        ax = axes[0][n]
        sub = combined_df[combined_df.user_budget == ub]
        sns.lineplot(ax=ax, data=sub, x='discount_factor', y='effective_slate_size',
                     hue='algorithm', hue_order=algo_order, palette=palette,
                     marker='o', errorbar='se')
        ax.set_title(f'Budget = {ub}')
        ax.set_xlabel('Discount Factor')
        ax.set_ylabel('Effective Slate Size' if n == 0 else '')
        if n > 0:
            ax.get_legend().remove()
    axes[0][-1].legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.suptitle('Effective Slate Size vs Discount Factor', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("../outputs/slatesize_v_budget_all.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Delta metrics: % change from discount_factor=0.2 to 0.8 ---
    df_02 = combined_df[combined_df.discount_factor == 0.2].copy()
    df_08 = combined_df[combined_df.discount_factor == 0.8].copy()
    merged = pd.merge(df_02, df_08, on=['user_budget', 'algorithm'], suffixes=('_02', '_08'))
    merged['delta_play_rate'] = (
        100. * (merged['play_rate_08'] - merged['play_rate_02']) / merged['play_rate_02']
    )
    merged['delta_effective_slate_size'] = (
        100. * (merged['effective_slate_size_08'] - merged['effective_slate_size_02'])
        / merged['effective_slate_size_02']
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=merged, x='user_budget', y='delta_effective_slate_size',
                 hue='algorithm', hue_order=algo_order, palette=palette, marker='o')
    plt.title('Δ Effective Slate Size (%, DF 0.2→0.8)')
    plt.xlabel('User Budget')
    plt.ylabel('Δ Effective Slate Size (%)')
    plt.tight_layout()
    plt.savefig("../outputs/delss_v_budget_all.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=merged, x='user_budget', y='delta_play_rate',
                 hue='algorithm', hue_order=algo_order, palette=palette, marker='o')
    plt.title('Δ Play Rate (%, DF 0.2→0.8)')
    plt.xlabel('User Budget')
    plt.ylabel('Δ Play Rate (%)')
    plt.tight_layout()
    plt.savefig("../outputs/dpr_v_budget_all.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison():
    """Detailed comparison plots: SARSA only (legacy), then all-5-algorithm overlays."""
    sarsa_results_df = json_to_dataframe("../outputs/results_sarsa.json")
    qlearning_results_df = json_to_dataframe("../outputs/results_qlearning.json")

    sns.set_theme('notebook')
    algo_order = ['SARSA', 'Q-Learning', 'PPO', 'GRPO', 'GSPO']
    palette = sns.color_palette("tab10", n_colors=len(algo_order))

    user_budgets = [150, 300, 500]

    # Legacy SARSA-only panels
    f, axes = plt.subplots(nrows=1, ncols=len(user_budgets), figsize=(16, 6), sharex=True, sharey=True)
    for n, ub in enumerate(user_budgets):
        sub_result = sarsa_results_df[sarsa_results_df.user_budget == ub].reset_index()
        sns.lineplot(ax=axes[n], data=sub_result,
                     x='discount_factor', y='play_rate', marker='o').set(title=f'User Budget: {ub}')

    f, axes = plt.subplots(nrows=1, ncols=len(user_budgets), figsize=(16, 6), sharex=True, sharey=True)
    for n, ub in enumerate(user_budgets):
        sub_result = sarsa_results_df[sarsa_results_df.user_budget == ub].reset_index()
        sns.lineplot(ax=axes[n], data=sub_result,
                     x='discount_factor', y='effective_slate_size', marker='o').set(title=f'User Budget: {ub}')

    sarsa_df_02 = sarsa_results_df[sarsa_results_df.discount_factor == 0.2].reset_index()
    sarsa_df_08 = sarsa_results_df[sarsa_results_df.discount_factor == 0.8].reset_index()
    cb_vs_rl_df = pd.merge(sarsa_df_02, sarsa_df_08, on=['user_budget', 'seed'])
    cb_vs_rl_df['delta_play_rate'] = (
        100. * (cb_vs_rl_df['play_rate_y'] - cb_vs_rl_df['play_rate_x']) / cb_vs_rl_df['play_rate_x']
    )
    cb_vs_rl_df['delta_effective_slate_size'] = (
        100. * (cb_vs_rl_df['effective_slate_size_y'] - cb_vs_rl_df['effective_slate_size_x'])
        / cb_vs_rl_df['effective_slate_size_x']
    )
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=cb_vs_rl_df, x='user_budget', y='delta_effective_slate_size', marker='o')
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=cb_vs_rl_df, x='user_budget', y='delta_play_rate', marker='o')

    ql_df_08 = qlearning_results_df[qlearning_results_df.discount_factor == 0.8]
    ql_vs_sarsa_df = pd.merge(ql_df_08, sarsa_df_08, on=['user_budget', 'seed'])
    ql_vs_sarsa_df['delta_play_rate'] = (
        100. * (ql_vs_sarsa_df['play_rate_x'] - ql_vs_sarsa_df['play_rate_y']) / ql_vs_sarsa_df['play_rate_y']
    )
    ql_vs_sarsa_df['delta_effective_slate_size'] = (
        100. * (ql_vs_sarsa_df['effective_slate_size_x'] - ql_vs_sarsa_df['effective_slate_size_y'])
        / ql_vs_sarsa_df['effective_slate_size_y']
    )
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=ql_vs_sarsa_df, x='user_budget', y='delta_effective_slate_size', marker='o')
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=ql_vs_sarsa_df, x='user_budget', y='delta_play_rate', marker='o')

    # --- All-5-algorithm overlay panels (3 focal budgets) ---
    combined_df = _load_all_results()
    sub_combined = combined_df[combined_df.user_budget.isin(user_budgets)]

    f, axes = plt.subplots(nrows=1, ncols=len(user_budgets),
                           figsize=(16, 6), sharex=True, sharey=True)
    for n, ub in enumerate(user_budgets):
        sub = sub_combined[sub_combined.user_budget == ub]
        sns.lineplot(ax=axes[n], data=sub, x='discount_factor', y='play_rate',
                     hue='algorithm', hue_order=algo_order, palette=palette, marker='o')
        axes[n].set_title(f'Budget = {ub}')
        axes[n].set_xlabel('Discount Factor')
        axes[n].set_ylabel('Play Rate' if n == 0 else '')
        if n < len(user_budgets) - 1:
            axes[n].get_legend().remove()
    plt.suptitle('Play Rate – All Algorithms', fontsize=13)
    plt.tight_layout()
    plt.savefig("../outputs/playrate_all_algos.png", dpi=300, bbox_inches='tight')
    plt.show()

    f, axes = plt.subplots(nrows=1, ncols=len(user_budgets),
                           figsize=(16, 6), sharex=True, sharey=True)
    for n, ub in enumerate(user_budgets):
        sub = sub_combined[sub_combined.user_budget == ub]
        sns.lineplot(ax=axes[n], data=sub, x='discount_factor', y='effective_slate_size',
                     hue='algorithm', hue_order=algo_order, palette=palette, marker='o')
        axes[n].set_title(f'Budget = {ub}')
        axes[n].set_xlabel('Discount Factor')
        axes[n].set_ylabel('Effective Slate Size' if n == 0 else '')
        if n < len(user_budgets) - 1:
            axes[n].get_legend().remove()
    plt.suptitle('Effective Slate Size – All Algorithms', fontsize=13)
    plt.tight_layout()
    plt.savefig("../outputs/slatesize_all_algos.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================
# Neural Network Policy Components (PPO and GRPO)
# ============================================================

class PolicyNetwork(nn.Module):
    """Scores each (state, item) pair to form a distribution over items.

    Input features: (row_position, relevance, cost, budget_to_go, cum_relevance).
    """
    def __init__(self, state_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ValueNetwork(nn.Module):
    """Estimates V(s) where s = (row_position, budget_to_go, cum_relevance)."""
    def __init__(self, state_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _build_item_features_tensor(k, relevance_all_items, cost_all_items,
                                 budget_to_go, cum_relevance, num_docs):
    arr = generate_feature_matrix(k, relevance_all_items, cost_all_items,
                                  budget_to_go, cum_relevance, num_docs)
    return torch.FloatTensor(arr)


def _sampled_action_logprob(policy_net, features, chosen_idx, excluded_set,
                             device, num_neg=512):
    """Approximate log prob of chosen_idx using sampled softmax over negatives."""
    num_docs = features.shape[0]
    valid = np.array([i for i in range(num_docs) if i not in excluded_set])
    k = min(num_neg, len(valid))
    sample_ids = np.random.choice(valid, k, replace=False)
    if chosen_idx not in sample_ids:
        sample_ids = np.append(sample_ids, chosen_idx)

    sample_feat = features[sample_ids].to(device)
    logits = policy_net(sample_feat)
    log_probs = torch.log_softmax(logits, dim=0)
    chosen_pos = int(np.where(sample_ids == chosen_idx)[0][0])
    return log_probs[chosen_pos]


def _generate_user_slate_neural(policy_net, all_user_budgets, relevance_all_items,
                                 cost_all_items, slate_size, num_docs, user_ind,
                                 device, deterministic=False):
    """Generate one user's slate via neural policy; return slate, log_probs, state_feats."""
    chosen, log_probs_list, state_feats_list = [], [], []
    budget = float(all_user_budgets[user_ind])
    cum_rel = float(np.exp(0.0))

    policy_net.eval()
    with torch.no_grad():
        for k in range(slate_size):
            feats = _build_item_features_tensor(
                k, relevance_all_items, cost_all_items, budget, cum_rel, num_docs
            ).to(device)
            logits = policy_net(feats)
            mask = torch.ones(num_docs, dtype=torch.bool, device=device)
            for a in chosen:
                mask[a] = False
            logits = logits.masked_fill(~mask, float('-inf'))

            if deterministic:
                action = logits.argmax().item()
            else:
                probs = torch.softmax(logits, dim=0)
                action = torch.multinomial(probs, 1).item()

            log_probs_list.append(torch.log_softmax(logits, dim=0)[action].item())
            state_feats_list.append([float(k), budget, cum_rel])
            chosen.append(action)
            budget -= float(cost_all_items[action])
            cum_rel += float(np.exp(relevance_all_items[action]))

    return chosen, log_probs_list, state_feats_list


def _collect_trajectories(policy_net, all_user_budgets, relevance_all_items,
                           cost_all_items, slate_size, num_docs, num_users, device):
    """Collect slates + metadata from all users under current policy."""
    all_slates = np.zeros((num_users, slate_size), dtype=np.int32)
    all_log_probs, all_state_feats = [], []
    for u in range(num_users):
        slate, lps, sfs = _generate_user_slate_neural(
            policy_net, all_user_budgets, relevance_all_items, cost_all_items,
            slate_size, num_docs, u, device
        )
        all_slates[u] = slate
        all_log_probs.append(lps)
        all_state_feats.append(sfs)
    return all_slates, all_log_probs, all_state_feats


def _get_per_position_rewards(traj, num_users, slate_size):
    """Per-user, per-position click reward (1 if clicked, else 0)."""
    choice = traj['user_response']['choice'].numpy()
    return [[1.0 if choice[u] == k else 0.0 for k in range(slate_size)]
            for u in range(num_users)]


def _compute_gae(rewards, values, discount_factor, gae_lambda=0.95):
    """Generalised Advantage Estimation for a single trajectory."""
    T = len(rewards)
    returns, advantages, gae = [0.0] * T, [0.0] * T, 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + discount_factor * next_v - values[t]
        gae = delta + discount_factor * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return returns, advantages


# ============================================================
# PPO
# ============================================================

def run_ppo(
        num_users, slate_size, num_docs, user_initial_budget,
        doc_costs, doc_relevances, discount_factor,
        num_iter=5, seed=0, ppo_epochs=3, clip_eps=0.2,
        lr=3e-4, entropy_coef=0.01, num_neg=512, device=None
):
    """Proximal Policy Optimisation for budget-constrained slate recommendation."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_net = PolicyNetwork(state_dim=5, hidden_dim=64).to(device)
    value_net = ValueNetwork(state_dim=3, hidden_dim=64).to(device)
    pol_opt = optim.Adam(policy_net.parameters(), lr=lr)
    val_opt = optim.Adam(value_net.parameters(), lr=lr)

    results = []
    for it in range(num_iter):
        print(f"PPO Iteration {it+1}/{num_iter}: collecting trajectories ...")
        all_slates, old_log_probs, all_state_feats = _collect_trajectories(
            policy_net, user_initial_budget, doc_relevances, doc_costs,
            slate_size, num_docs, num_users, device
        )
        traj = get_simulation_data(
            num_users=num_users, slate_size=slate_size, num_docs=num_docs,
            user_initial_budget=user_initial_budget, doc_costs=doc_costs,
            doc_relevances=doc_relevances, user_slates=all_slates,
            default_no_choice_logit=4.0, seed=seed
        )
        metrics = compute_metrics(traj, slate_size, num_users)

        per_pos_rewards = _get_per_position_rewards(traj, num_users, slate_size)

        # Compute GAE for all users
        all_returns, all_advantages = [], []
        for u in range(num_users):
            sf = torch.FloatTensor(all_state_feats[u]).to(device)
            with torch.no_grad():
                vals = value_net(sf).cpu().numpy().tolist()
            rets, advs = _compute_gae(per_pos_rewards[u], vals, discount_factor)
            all_returns.append(rets)
            all_advantages.append(advs)

        flat_adv = np.array([a for advs in all_advantages for a in advs])
        adv_mean, adv_std = flat_adv.mean(), flat_adv.std() + 1e-8

        policy_net.train()
        value_net.train()
        for _ in range(ppo_epochs):
            for u in range(num_users):
                budget = float(user_initial_budget[u])
                cum_rel = float(np.exp(0.0))
                chosen_so_far = []
                for k in range(slate_size):
                    chosen = int(all_slates[u, k])
                    old_lp = old_log_probs[u][k]
                    ret = all_returns[u][k]
                    adv = (all_advantages[u][k] - adv_mean) / adv_std

                    sf = torch.FloatTensor([[k, budget, cum_rel]]).to(device)
                    val_pred = value_net(sf)
                    val_loss = (val_pred - ret) ** 2
                    val_opt.zero_grad()
                    val_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                    val_opt.step()

                    feats = _build_item_features_tensor(
                        k, doc_relevances, doc_costs, budget, cum_rel, num_docs
                    ).to(device)
                    new_lp = _sampled_action_logprob(
                        policy_net, feats, chosen, set(chosen_so_far), device, num_neg
                    )
                    ratio = torch.exp(new_lp - old_lp)
                    surr = torch.min(ratio * adv,
                                     torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)

                    # Entropy over sampled subset for regularisation
                    valid = [i for i in range(num_docs) if i not in set(chosen_so_far)]
                    samp = np.random.choice(valid, min(num_neg, len(valid)), replace=False)
                    ent_probs = torch.softmax(policy_net(feats[samp].to(device)), dim=0)
                    entropy = -(ent_probs * (ent_probs + 1e-8).log()).sum()

                    pol_loss = -surr - entropy_coef * entropy
                    pol_opt.zero_grad()
                    pol_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                    pol_opt.step()

                    chosen_so_far.append(chosen)
                    budget -= float(doc_costs[chosen])
                    cum_rel += float(np.exp(doc_relevances[chosen]))

        print(f"PPO Iteration {it+1}/{num_iter}: Play Rate={metrics[0]:.4f}, Avg Imp={metrics[1]:.4f}")
        results.append(metrics)

    return results, policy_net


# ============================================================
# GRPO
# ============================================================

def run_grpo(
        num_users, slate_size, num_docs, user_initial_budget,
        doc_costs, doc_relevances, discount_factor,
        num_iter=5, seed=0, G=4, lr=3e-4,
        num_neg=512, device=None
):
    """Group Relative Policy Optimisation for budget-constrained slate recommendation.

    Samples G slates per iteration, normalises rewards within the group, and
    updates the policy proportional to the group-relative advantage — no critic needed.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_net = PolicyNetwork(state_dim=5, hidden_dim=64).to(device)
    pol_opt = optim.Adam(policy_net.parameters(), lr=lr)

    results = []
    for it in range(num_iter):
        print(f"GRPO Iteration {it+1}/{num_iter}: sampling {G} trajectory groups ...")

        group_slates, group_log_probs, group_rewards = [], [], []
        for g in range(G):
            slates, lps, _ = _collect_trajectories(
                policy_net, user_initial_budget, doc_relevances, doc_costs,
                slate_size, num_docs, num_users, device
            )
            traj = get_simulation_data(
                num_users=num_users, slate_size=slate_size, num_docs=num_docs,
                user_initial_budget=user_initial_budget, doc_costs=doc_costs,
                doc_relevances=doc_relevances, user_slates=slates,
                default_no_choice_logit=4.0, seed=seed + g
            )
            # Terminal per-user reward: 1 if user played anything
            choice = traj['user_response']['choice'].numpy()
            play_flags = (choice != slate_size).astype(np.float32)
            group_slates.append(slates)
            group_log_probs.append(lps)
            group_rewards.append(play_flags)

        metrics = compute_metrics(traj, slate_size, num_users)  # last group

        # Normalise rewards within group dimension
        rewards_arr = np.stack(group_rewards, axis=0)  # (G, num_users)
        r_mean = rewards_arr.mean(axis=0)
        r_std = rewards_arr.std(axis=0) + 1e-8
        norm_rewards = (rewards_arr - r_mean) / r_std  # (G, num_users)

        policy_net.train()
        for g in range(G):
            for u in range(num_users):
                budget = float(user_initial_budget[u])
                cum_rel = float(np.exp(0.0))
                chosen_so_far = []
                nr = float(norm_rewards[g, u])
                for k in range(slate_size):
                    chosen = int(group_slates[g][u, k])
                    feats = _build_item_features_tensor(
                        k, doc_relevances, doc_costs, budget, cum_rel, num_docs
                    ).to(device)
                    new_lp = _sampled_action_logprob(
                        policy_net, feats, chosen, set(chosen_so_far), device, num_neg
                    )
                    loss = -new_lp * nr
                    pol_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                    pol_opt.step()

                    chosen_so_far.append(chosen)
                    budget -= float(doc_costs[chosen])
                    cum_rel += float(np.exp(doc_relevances[chosen]))

        print(f"GRPO Iteration {it+1}/{num_iter}: Play Rate={metrics[0]:.4f}, Avg Imp={metrics[1]:.4f}")
        results.append(metrics)

    return results, policy_net


# ============================================================
# GSPO – Group Sequence Policy Optimization
# ============================================================

def run_gspo(
        num_users, slate_size, num_docs, user_initial_budget,
        doc_costs, doc_relevances, discount_factor,
        num_iter=5, seed=0, G=4, lr=3e-4,
        clip_eps=0.2, num_neg=512, device=None
):
    """Group Sequence Policy Optimization (GSPO) for budget-constrained slate recommendation.

    Like GRPO, G slates are sampled per iteration and group-relative rewards are used
    as advantages (no critic).  Unlike GRPO, the importance-sampling ratio is computed
    at the *sequence* level — the product of per-step ratios over the full slate — and
    clipping is applied to that sequence-level ratio before the policy update.
    This avoids the compounding-ratio instability of per-step IS while preserving the
    critic-free simplicity of group-relative normalisation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_net = PolicyNetwork(state_dim=5, hidden_dim=64).to(device)
    pol_opt = optim.Adam(policy_net.parameters(), lr=lr)

    results = []
    for it in range(num_iter):
        print(f"GSPO Iteration {it+1}/{num_iter}: sampling {G} trajectory groups ...")

        group_slates, group_old_log_probs, group_rewards = [], [], []
        for g in range(G):
            slates, lps, _ = _collect_trajectories(
                policy_net, user_initial_budget, doc_relevances, doc_costs,
                slate_size, num_docs, num_users, device
            )
            traj = get_simulation_data(
                num_users=num_users, slate_size=slate_size, num_docs=num_docs,
                user_initial_budget=user_initial_budget, doc_costs=doc_costs,
                doc_relevances=doc_relevances, user_slates=slates,
                default_no_choice_logit=4.0, seed=seed + g
            )
            # Terminal per-user reward: 1 if any item was chosen, else 0
            choice = traj['user_response']['choice'].numpy()
            play_flags = (choice != slate_size).astype(np.float32)
            group_slates.append(slates)
            group_old_log_probs.append(lps)
            group_rewards.append(play_flags)

        # Record metrics from the final group for logging
        metrics = compute_metrics(traj, slate_size, num_users)

        # Group-relative reward normalisation
        rewards_arr = np.stack(group_rewards, axis=0)  # (G, num_users)
        r_mean = rewards_arr.mean(axis=0)
        r_std = rewards_arr.std(axis=0) + 1e-8
        norm_rewards = (rewards_arr - r_mean) / r_std  # (G, num_users)

        # GSPO policy update — sequence-level IS ratio with clipping
        policy_net.train()
        for g in range(G):
            for u in range(num_users):
                budget = float(user_initial_budget[u])
                cum_rel = float(np.exp(0.0))
                chosen_so_far = []
                nr = float(norm_rewards[g, u])

                # Accumulate per-step log ratios into a sequence-level log ratio
                seq_log_ratio = torch.zeros(1, device=device)
                step_new_lps = []
                for k in range(slate_size):
                    chosen = int(group_slates[g][u, k])
                    old_lp = group_old_log_probs[g][u][k]

                    feats = _build_item_features_tensor(
                        k, doc_relevances, doc_costs, budget, cum_rel, num_docs
                    ).to(device)
                    new_lp = _sampled_action_logprob(
                        policy_net, feats, chosen, set(chosen_so_far), device, num_neg
                    )
                    step_new_lps.append(new_lp)
                    seq_log_ratio = seq_log_ratio + (new_lp - old_lp)

                    chosen_so_far.append(chosen)
                    budget -= float(doc_costs[chosen])
                    cum_rel += float(np.exp(doc_relevances[chosen]))

                # Sequence-level clipped surrogate objective
                seq_ratio = torch.exp(seq_log_ratio)
                surr1 = seq_ratio * nr
                surr2 = torch.clamp(seq_ratio, 1 - clip_eps, 1 + clip_eps) * nr
                loss = -torch.min(surr1, surr2)

                pol_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                pol_opt.step()

        print(f"GSPO Iteration {it+1}/{num_iter}: Play Rate={metrics[0]:.4f}, Avg Imp={metrics[1]:.4f}")
        results.append(metrics)

    return results, policy_net


# ============================================================
# Seed-level runners (mirror existing run_sarsa_for_seed style)
# ============================================================

def run_ppo_for_seed(budget, discount_factor, seed, num_iter, num_users, item_relevances):
    print(f"====== PPO Experiment Start - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    np.random.seed(seed)
    params = Params(
        num_users=num_users, num_docs=num_items, slate_size=slate_size,
        epsilon=epsilon, budget=budget, seed=seed, discount_factor=discount_factor
    )
    all_user_budgets = np.random.lognormal(np.log(params.budget), user_budget_scale, params.num_users)
    cost_all_items = np.random.uniform(cost_low, cost_high, size=params.num_docs)
    results, _ = run_ppo(
        num_users=params.num_users, slate_size=params.slate_size, num_docs=params.num_docs,
        user_initial_budget=all_user_budgets, doc_costs=cost_all_items,
        doc_relevances=item_relevances, discount_factor=params.discount_factor,
        num_iter=num_iter, seed=params.seed
    )
    print(f"====== PPO Experiment End - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    return params, results


def run_grpo_for_seed(budget, discount_factor, seed, num_iter, num_users, item_relevances):
    print(f"====== GRPO Experiment Start - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    np.random.seed(seed)
    params = Params(
        num_users=num_users, num_docs=num_items, slate_size=slate_size,
        epsilon=epsilon, budget=budget, seed=seed, discount_factor=discount_factor
    )
    all_user_budgets = np.random.lognormal(np.log(params.budget), user_budget_scale, params.num_users)
    cost_all_items = np.random.uniform(cost_low, cost_high, size=params.num_docs)
    results, _ = run_grpo(
        num_users=params.num_users, slate_size=params.slate_size, num_docs=params.num_docs,
        user_initial_budget=all_user_budgets, doc_costs=cost_all_items,
        doc_relevances=item_relevances, discount_factor=params.discount_factor,
        num_iter=num_iter, seed=params.seed
    )
    print(f"====== GRPO Experiment End - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    return params, results


def run_gspo_for_seed(budget, discount_factor, seed, num_iter, num_users, item_relevances):
    print(f"====== GSPO Experiment Start - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    np.random.seed(seed)
    params = Params(
        num_users=num_users, num_docs=num_items, slate_size=slate_size,
        epsilon=epsilon, budget=budget, seed=seed, discount_factor=discount_factor
    )
    all_user_budgets = np.random.lognormal(np.log(params.budget), user_budget_scale, params.num_users)
    cost_all_items = np.random.uniform(cost_low, cost_high, size=params.num_docs)
    results, _ = run_gspo(
        num_users=params.num_users, slate_size=params.slate_size, num_docs=params.num_docs,
        user_initial_budget=all_user_budgets, doc_costs=cost_all_items,
        doc_relevances=item_relevances, discount_factor=params.discount_factor,
        num_iter=num_iter, seed=params.seed
    )
    print(f"====== GSPO Experiment End - [Seed: {seed}, Budget: {budget}, Discount Factor: {discount_factor}] ======")
    return params, results


def get_item_relevances():
    item_relevance_map = {}
    CONFIG = Config()
    dataset = CSVDataGenerator(CONFIG.train_set, CONFIG.seq_len)
    preds = predict(CONFIG, 'train')
    preds = preds.reshape(-1)
    for idx, item in enumerate(dataset.items):
        if item not in item_relevance_map:
            item_relevance_map[item] = [preds[idx]]
        else:
            item_relevance_map[item].append(preds[idx])
    # print(len(item_relevance_map))

    item_relevances = [np.median(values) for values in item_relevance_map.values()]
    item_relevances = np.array(item_relevances)
    # print(item_relevances.shape)
    # print(item_relevances[:5])
    return item_relevances

if __name__ == "__main__":
    # FIXED Hyper-parameters used in the results in the notebook
    num_items = 142998
    slate_size = 30
    epsilon = 0.1
    user_budget_scale = 0.5
    cost_low = 0.
    cost_high = 100.
    num_users = 150

    parameter_range = {
        'budget': [100., 150., 200.0, 250., 300., 400., 500.],
        'discount_factor': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'seed': range(5)
    }
    # SUGGESTION: first try out a few samples from the grid
    # parameter_grid = ParameterSampler(parameter_range, 1)
    parameter_grid = ParameterGrid(parameter_range)

    item_relevances = get_item_relevances()

    # ---- SARSA ----
    results_sarsa = {}
    for p in parameter_grid:
        params, experiment_results = run_sarsa_for_seed(
            seed=p['seed'], budget=p['budget'],
            discount_factor=p['discount_factor'],
            num_iterations_for_model_fitting=5,
            num_users=num_users,
            item_relevances=item_relevances
        )
        results_sarsa[str(params)] = experiment_results

    with open('../outputs/results_sarsa.json', 'w') as f:
        json.dump(results_sarsa, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # ---- Q-Learning ----
    results_qlearning = {}
    for p in parameter_grid:
        params, experiment_results = run_qlearning_for_seed(
            seed=p['seed'],
            budget=p['budget'],
            discount_factor=p['discount_factor'],
            num_iter=5,
            num_users=num_users,
            item_relevances=item_relevances
        )
        results_qlearning[str(params)] = experiment_results

    with open('../outputs/results_qlearning.json', 'w') as f:
        json.dump(results_qlearning, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # ---- PPO ----
    results_ppo = {}
    for p in parameter_grid:
        params, experiment_results = run_ppo_for_seed(
            seed=p['seed'],
            budget=p['budget'],
            discount_factor=p['discount_factor'],
            num_iter=5,
            num_users=num_users,
            item_relevances=item_relevances
        )
        results_ppo[str(params)] = experiment_results

    with open('../outputs/results_ppo.json', 'w') as f:
        json.dump(results_ppo, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # ---- GRPO ----
    results_grpo = {}
    for p in parameter_grid:
        params, experiment_results = run_grpo_for_seed(
            seed=p['seed'],
            budget=p['budget'],
            discount_factor=p['discount_factor'],
            num_iter=5,
            num_users=num_users,
            item_relevances=item_relevances
        )
        results_grpo[str(params)] = experiment_results

    with open('../outputs/results_grpo.json', 'w') as f:
        json.dump(results_grpo, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # ---- GSPO ----
    results_gspo = {}
    for p in parameter_grid:
        params, experiment_results = run_gspo_for_seed(
            seed=p['seed'],
            budget=p['budget'],
            discount_factor=p['discount_factor'],
            num_iter=5,
            num_users=num_users,
            item_relevances=item_relevances
        )
        results_gspo[str(params)] = experiment_results

    with open('../outputs/results_gspo.json', 'w') as f:
        json.dump(results_gspo, f, indent=4, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # ---- Plotting ----
    plot_results()
    plot_comparison()