import numpy as np
import scipy.optimize as so
import pandas as pd
import pgmpy.models as pm
import pgmpy.factors.discrete as pfd
import pgmpy.inference as pi

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def get_demographic_data():
    """
    Process estimated 2020 demographic data from: https://wonder.cdc.gov/bridged-race-v2020.html
    """
    df = pd.read_csv('BRPE2020.txt', sep='\t', header=0)
    df.rename(columns={'Gender Code': 'Sex', 'Age Group Code': 'AGC', 'Race': 'RE', 'Population': 'Pop'}, inplace=True)

    df.loc[(df.Ethnicity == 'Hispanic or Latino'), 'RE'] = 'H'
    rs = (('American Indian or Alaska Native', 'N'), ('Asian or Pacific Islander', 'A'), ('Black or African American', 'B'),
          ('White', 'W'))
    for r in rs:
        df.loc[(df.RE == r[0]), 'RE'] = r[1]
    age_mask = []
    for row in df.itertuples():
        try:
            if int(row.AGC.split('-')[0].rstrip('+')) < 65:
                age_mask.append('Y')
            else:
                age_mask.append('O')
        except AttributeError:
            age_mask.append(None)
    df['Age'] = age_mask
    df.drop(columns=['Gender', 'Age Group', 'Race Code', 'Ethnicity Code', 'Ethnicity', 'AGC'], inplace=True)
    df = df[df['Notes'] != 'Total']
    return df


def process_covid_case_data():
    """
    Process COVID data from:
    https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4
    """
    df = pd.read_csv('COVID_01102021.csv', use_cols=['sex', 'race', 'ethnicity', 'age_group', 'hosp_yn', 'death_yn'])
    print('Before', len(df))
    # df.drop(columns=['case_month', 'res_state', 'state_fips_code', 'res_county', 'county_fips_code',
    #                   'case_positive_specimen_interval', 'case_onset_interval', 'process', 'exposure_yn', 'current_status',
    #                   'symptom_status', 'icu_yn', 'underlying_conditions_yn'], inplace=True)
    df.dropna(inplace=True)
    df = df[df.race.isin(['Missing', 'Unknown']) == False]
    df = df[df.ethnicity.isin(['Missing', 'Unknown']) == False]
    df = df[df.hosp_yn.isin(['Missing', 'Unknown']) == False]
    df = df[df.death_yn.isin(['Missing', 'Unknown']) == False]
    df.rename(columns={'sex': 'Sex', 'age_group': 'AGC', 'race': 'RE'}, inplace=True)
    df = df[df.AGC.isin(['Missing', 'Unknown']) == False]
    df.loc[(df.ethnicity == 'Hispanic/Latino'), 'RE'] = 'H'
    rs = (('American Indian/Alaska Native', 'N'), ('Asian', 'A'), ('Black', 'B'), ('White', 'W'),
          ('Native Hawaiian/Other Pacific Islander', 'A'))
    for r in rs:
        df.loc[(df.RE == r[0]), 'RE'] = r[1]
    df = df[df.RE.isin(['Multiple/Other']) == False]
    df.loc[(df.Sex == 'Male'), 'Sex'] = 'M'
    df.loc[(df.Sex == 'Female'), 'Sex'] = 'F'
    df.loc[(df.hosp_yn == 'No'), 'hosp_yn'] = 0
    df.loc[(df.hosp_yn == 'Yes'), 'hosp_yn'] = 1
    df.loc[(df.death_yn == 'No'), 'death_yn'] = 0
    df.loc[(df.death_yn == 'Yes'), 'death_yn'] = 1
    age_mask = []
    for row in df.itertuples():
        try:
            if int(row.AGC.split()[0].rstrip('+')) < 65:
                age_mask.append('Y')
            else:
                age_mask.append('O')
        except AttributeError:
            age_mask.append(None)
    df['Age'] = age_mask
    df.drop(columns=['ethnicity', 'AGC'], inplace=True)
    df.to_csv('cleaned_covid.csv')
    print('After', len(df))
    return df


def get_covid_case_data():
    return pd.read_csv('cleaned_covid.csv')


demo = get_demographic_data()
print(demo.head())
# case = get_covid_case_data()


def compute_age():
    sub_pop = pd.concat([demo.groupby(['Sex', 'RE']).sum()])
    age = round(demo.groupby(['Age', 'Sex', 'RE']).sum() / sub_pop, 3).values.tolist()
    old = [i[0] for i in age[::2]]
    young = [i[0] for i in age[1::2]]
    return [old, young]


def compute_vaccine():
    """
    Uses Covid demographic data from: https://covid.cdc.gov/covid-data-tracker/#vaccination-demographic
    """
    df = pd.read_csv('COVID_demo_31102021.csv', usecols=['Date', 'Demographic_category', 'Series_Complete_Yes',
                                                         'Series_Complete_Pop_Pct_known'])
    df = df[df['Date'] == '10/31/2021']
    demo = {}
    for row in df.itertuples():
        if row[2] == 'Age_known':
            demo[row[2]] = row[3]
        elif row[2].startswith('Ages') and row[2] not in ('Ages_<18yrs', 'Ages_30-39_yrs', 'Ages_18-29_yrs'):
            demo[row[2]] = row[4] / 100
        elif row[2].startswith('Race'):
            re = row[2].lstrip('Race_eth_')
            if re in ('NHAsian', 'NHNHOPI'):
                re = 'NHAsian'
                if re in demo:
                    demo[re] += row[4] / 100
                else:
                    demo[re] = row[4] / 100
            elif re in ('NHBlack', 'Hispanic', 'NHAIAN', 'NHWhite'):
                demo[re] = row[4] / 100
    print(demo)

    oa, ob, oh, on, ow, ya, yb, yh, yn, yw = solve(total_vaccines, o_pct, y_pct, a_pct, b_pct, h_pct, n_pct, w_pct)
    a = oa + ya
    b = ob + yb
    h = oh + yh
    n = on + yn
    w = ow + yw
    solution = [round(i, 3) for i in (oa/a, ob/b, oh/h, on/n, ow/w, ya/a, yb/b, yh/h, yn/n, yw/w)]
    return [solution[:5], solution[5:]]


def solve(total, pct_o, pct_y, pct_a, pct_b, pct_h, pct_n, pct_w):

    obj = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    lhs_ineq = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
    rhs_ineq = [int(total * pct_o),
                int(total * pct_y),
                int(total * pct_a),
                int(total * pct_b),
                int(total * pct_h),
                int(total * pct_n),
                int(total * pct_w)]

    ares = demo.groupby(['Age', 'RE']).sum().values
    bnds = []
    for i in range(10):
        bnds.append((0.0, ares[i][0]))

    solution = so.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnds, method='revised simplex')
    return solution.x


pop = demo['Pop'].sum()
p_sex = round(demo.groupby('Sex').sum() / pop, 3).values.tolist()  # 0: F, 1: M
p_re = round(demo.groupby('RE').sum() / pop, 3).values.tolist()  # 0: A, 1: B, 2: H, 3: N, 4: W
p_age = compute_age()  # 0: O, 1: Y
p_covid = None
p_vaccine = compute_vaccine()
p_outcome = None
print(p_vaccine)
print(p_age)

assert False

# covid_model = pm.BayesianNetwork([('Sex', 'Age'), ('Sex', 'Vaccine'),
#                                   ('RE', 'Age'), ('RE', 'Covid'), ('RE', 'Outcome'), ('RE', 'Vaccine'),
#                                   ('Age', 'Covid'), ('Age', 'Outcome'), ('Age', 'Vaccine'),
#                                   ('Covid', 'Outcome'),
#                                   ('Vaccine', 'Covid'), ('Vaccine', 'Outcome')])

d_model = pm.BayesianNetwork([('Sex', 'Age'), ('RE', 'Age')])

cpd_sex = pfd.TabularCPD(variable='Sex', variable_card=2, values=p_sex)
cpd_re = pfd.TabularCPD(variable='RE', variable_card=5, values=p_re)
cpd_age = pfd.TabularCPD(variable='Age', variable_card=2, values=p_age, evidence=['Sex', 'RE'], evidence_card=[2, 5])

d_model.add_cpds(cpd_sex, cpd_re, cpd_age)
print(d_model.check_model())
d_infer = pi.VariableElimination(d_model)

q = d_infer.query(variables=['Sex'], evidence={'RE': 2, 'Age': 1})
print(q)
