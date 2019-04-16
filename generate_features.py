import pandas as pd
import os

import numpy as np
from pandas import read_csv
from datetime import datetime, timedelta
from timeit import default_timer


def features():
    features_type = {
        'H_RD': int,  # Rest days
        # 'H_WR': float, # Win ratio for last 10 games
        'H_WLG': bool,  # Won last game
        'H_EFG': float,  # Effective field goal %
        'H_OREB': float,  # Offesnie RB %
        'H_DREB': float,  # Defensive RB %
        'H_AST_TO': float,  # Assit to Turn over ratio
        # 'H_W': float,  # win percentage
        'A_RD': int,  # Rest days
        # 'A_WR': float, # Win ratio for last 10 games
        'A_WLG': bool,  # Won last game
        'A_EFG': float,  # Effective field goal %
        'A_OREB': float,  # Offesnie RB %
        'A_DREB': float,  # Defensive RB %
        'A_AST_TO': float,  # Assit to Turn over ratio
        # 'A_W': float  # win percentage
    }
    return features_type


def check_court(match):
    return AWAY if match.split()[1] == '@' else HOME


def check_winner(game_inf):
    court = check_court(game_inf['MATCHUP'])
    if court == HOME:
        winner = HOME if game_inf['WL'] == 'W' else AWAY
    else:
        winner = AWAY if game_inf['WL'] == 'W' else HOME

    return winner


def guess_starters(players_bx, game_inf):
    return players_bx[(players_bx.GAME_ID == game_inf['GAME_ID'])
                      & (players_bx.TEAM_ID == game_inf['TEAM_ID'])].head(
        NUM_PLAYERS).sort_values(
        by='MIN', ascending=False)  # TODO: clean this


def assign_features(sample, game_inf, teams_meta, player_stats):
    court_prefix = check_court(game_inf['MATCHUP']) + "_"
    game_id, team_id = game_inf['GAME_ID'], game_inf['TEAM_ID']
    game_date = datetime.strptime(game_inf['GAME_DATE'][:10], '%Y-%m-%d')
    weights = player_stats['MIN'] / (player_stats['MIN'].sum())

    sample.update({
        court_prefix + 'RD': (game_date - teams_meta[team_id]['last_game_dt']).days,
        court_prefix + 'WLG': teams_meta[team_id]['WLG'],
        court_prefix + 'EFG': (weights * player_stats['EFG_PCT']).sum(),
        court_prefix + 'OREB': (weights * player_stats['OREB_PCT']).sum(),
        court_prefix + 'DREB': (weights * player_stats['DREB_PCT']).sum(),
        court_prefix + 'AST_TO': (weights * player_stats['AST_TO']).sum(),
        court_prefix + 'W_PCT': (weights * player_stats['W_PCT']).sum()
    })


"""
DEFINITIONS
"""
HOME = 'H'
AWAY = 'A'


"""
SETTINGS 
"""
NUM_PLAYERS = 7  # Number of players used to generate features
REQ_PLAYER_STATS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "OREB_PCT",
    "DREB_PCT",
    "EFG_PCT",
    "AST_TO",
    "W_PCT",
    "MIN"
]


"""
SCRIPT
"""
abs_path = os.getcwd()
df_team_bx = read_csv(abs_path + '/data/team_boxscore_stats_1997-19.csv')
df_player_bx = read_csv(abs_path + '/data/player_boxscore_stats_1997-19_adv.csv')


start_time = default_timer()
x_train = list()
y_train = list()
season_id = None
incomplete_sample = dict()  # {'%ID": %i}

for i, game_inf in df_team_bx.iterrows():

    # Fetch last seasons player stats
    if season_id != game_inf['SEASON_ID']:  #TODO: Season 1999 splits regular/playoff
        season_id = game_inf['SEASON_ID']
        teams_meta = dict()
        yr = game_inf['GAME_DATE'][:4]
        file_path ='data/player_yearly_stats/player_stats_' + str(int(yr) - 1)[-2:] + '-' + yr[-2:] + '.csv'
        df_player_stats_hist = read_csv(file_path)
        print('Processing Season' + yr + '...')

    game_id, team_id = game_inf['GAME_ID'], game_inf['TEAM_ID']
    if not teams_meta.get(team_id):
        teams_meta[team_id] = {
            'last_game_dt': datetime.strptime(game_inf['GAME_DATE'][:10], '%Y-%m-%d'),
            'WLG': True if game_inf['WL'] == 'W' else False
        }
        continue

    # TODO: implement winning streaks
    # if len(teams_meta.get(team_id, {}).get(last_5_games, [])) < 5:
    #     team_meta[team_id]['last_5_games'] = team_meta[team_id].get('last_5_games',[]) + []
    #     continue

    if game_id in incomplete_sample:
        sample = x_train[incomplete_sample[game_id]]
        incomplete_sample.pop(game_id)
    else:
        sample = features()
        x_train.append(sample)
        y_train.append(check_winner(game_inf))
        incomplete_sample[game_id] = len(x_train) - 1

    starters = guess_starters(df_player_bx, game_inf)
    starters_stats = df_player_stats_hist.merge(starters, on='PLAYER_ID', how='right', suffixes=('', '_BX'))

    # TODO: Assign residual stats, deal with non-existent stats
    # if not avg_stats:
    #     # do not overwrite
    #     player_stats = df_player_stats_cur[df_player_stats_cur.PLAYER_ID == player['PLAYER_ID']]

    assign_features(sample, game_inf, teams_meta, starters_stats)
    teams_meta[team_id] = {
        'last_game_dt': datetime.strptime(game_inf['GAME_DATE'][:10], '%Y-%m-%d'),
        'WLG': True if game_inf['WL'] == 'W' else False
    }

print('Cleaning incomplete samples...')

# Clean throw away data
for i in sorted(list(incomplete_sample.values()), reverse=True):
    x_train.pop(i)
    y_train.pop(i)

end_time = default_timer()
print(len(x_train), 'Samples Generated in', timedelta(seconds=end_time - start_time))

add_info = NUM_PLAYERS + 'P_AVG'
add_info += '_' if add_info else ''
time_lbl = datetime.now().strftime('%m-%d-%H%M%S')

feat_file = abs_path + '/feature_data/x_train_' + add_info + time_lbl
lbl_file = abs_path + '/feature_data/y_train_' + add_info + time_lbl

np.save(feat_file, pd.DataFrame(x_train).values)
np.save(lbl_file, y_train)

print('Training data generated in feature_data folder')
