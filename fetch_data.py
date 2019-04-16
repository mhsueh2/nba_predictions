import requests
import numpy as np
import pandas as pd

S_STR = 1997
S_END = 2018

'''
Get game to game boxscore of each players from given date range
'''
print('Processing player data...')

file_name = 'data/player_boxscore_stats_' + str(S_STR) + '-' + str(S_END)[-2:] + '_adv.csv'
with open(file_name, 'w') as f:
    req_columns = [
            "SEASON_YEAR",
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "GAME_ID",
            "GAME_DATE",
            "MATCHUP",
            "WL",
            "MIN",
            "OREB_PCT",
            "DREB_PCT",
            "EFG_PCT",
            "AST_TO"
    ]
    for y in range(S_STR, S_END):
        print(y)
        season_param = str(y) + '-' + str(y + 1)[-2:]
        url = 'https://stats.nba.com/stats/playergamelogs/?LastNGame=0&LeagueID=00&MeasureType=Advanced&Month=0&OpponentTeamID=0&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlusMinus=N&Rank=N&SeasonType=Regular+Season&TeamID=0&TwoWay=0&PlayerPosition&GameScope&PlayerExperience&StarterBench=Starter&Outcome&Location&SeasonSegment&DateFrom&DateTo&VsConference&VsDivision&Gamesegment&LastNGames=0&Season='
        url = url + season_param
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

        response = requests.get(url, headers=headers)
        raw_data = response.json()

        data = raw_data['resultSets'][0]['rowSet']
        columns = raw_data['resultSets'][0]['headers']
        df = pd.DataFrame(data=np.array(data), columns=columns)[req_columns]

        if y == S_STR:
            df.to_csv(f, header=True, index=False)
        else:
            df.to_csv(f, header=False, index=False)
'''
Get game to game boxscore of each players team given date range
'''
# data = list()
# print('Processing team data...')
# for y in range(S_STR, S_END):
#     print(y)
#     season_param = str(y) + '-' + str(y + 1)[-2:]
#     url = 'https://stats.nba.com/stats/leaguegamelog/?LeagueID=00&Sorter=DATE&PlayerOrTeam=T&SeasonType=Regular Season&Direction=DESC&Season=' + season_param
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}
#
#     response = requests.get(url, headers=headers)
#     raw_data = response.json()
#
#     data.extend(raw_data['resultSets'][0]['rowSet'])
#
# columns = raw_data['resultSets'][0]['headers']
# np.array(data)
# df = pd.DataFrame(data=data, columns=columns)
#
# file_name = 'team_boxscore_stats_' + str(S_STR) + '-' + str(S_END)[-2:] + '.csv'
# df.to_csv('data/'+file_name, index=False)
#
# pd.DataFrame()
