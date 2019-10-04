import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ym2y(ym):
    """
    Convert "year month" to "year"

    :param ym: 'str'. format 'xxYxxM' or 'xxY' or 'xxM'
    :return: 'float'. year to maturity
    """
    if 'Y' not in ym:
        return int(ym[:-1]) / 12
    elif 'M' not in ym:
        return int(ym[:-1])
    else:
        return int(ym.split('Y')[0]) + int(ym.split('Y')[1][:-1]) / 12


def preproc(df, bond_type='G_N_A'):
    """
    preprocessing pipeline

    :param df: pandas.dataframe
    :param bond_type: str. include specified bonds. 'G_N_A' for AAA rated bonds, 'G_N_C' for all bonds.
    :return: pandas.dataframe. processed dataframe
    """
    df = df[df['INSTRUMENT_FM'].map(lambda x: x == bond_type)]
    # Only keep spot rates
    df = df[df['DATA_TYPE_FM'].str.startswith('SR_')]
    df['DATA_TYPE_FM'] = df['DATA_TYPE_FM'].map(lambda x: str(x)[3:]).map(ym2y)
    df = df.sort_values(by=['DATA_TYPE_FM'])
    df = df.drop(columns=['INSTRUMENT_FM'])
    return df.rename(columns={"DATA_TYPE_FM": "t2matur", "OBS_VALUE": "spot_rate"}).set_index('t2matur')


if __name__ == "__main__":
    df = pd.read_csv('data.csv',
                     usecols=['DATA_TYPE_FM', 'OBS_VALUE', 'INSTRUMENT_FM'],
                     )
    df_3a = preproc(df, bond_type='G_N_A')
    df_all = preproc(df, bond_type='G_N_C')

    plt.plot(df_3a.index.values, df_3a['spot_rate'], label='AAA rated bonds')
    plt.plot(df_all.index.values, df_all['spot_rate'], '--', label='All bonds')
    plt.legend()
    plt.title('Spots Rate vs. Years to Maturity')
    plt.xlabel('Residual maturity in years')
    plt.ylabel('Yield in %')
    plt.grid()
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig('p2.pdf')
    plt.show()
