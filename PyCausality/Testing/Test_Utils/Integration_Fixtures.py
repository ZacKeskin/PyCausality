fixtures = {'test1':
                    {'bins': {'S1': [95,99, 99.6, 100, 101,105], 
                              'S2': [95,99, 99.6, 100, 101,105]},
                     'S1': 100,
                     'S2': 100,
                     'T':  5,
                     'N':  100, 
                     'mu1': 0, 
                     'mu2': 0,
                     'sigma1': 0.01,
                     'sigma2': 0.01, 
                     'alpha': 0,
                     'lag': 1,
                     'seed': 10,
                     'expected_TE':0,
                     'expected_Z_score':0,
                     'tolerance':0.05

                    },
            'test2':
                    {'bins': {'S1': [95,99, 99.6, 100, 101,105], 
                              'S2': [95,99, 99.6, 100, 101,105]},
                     'S1': 100,
                     'S2': 100,
                     'T':  5,
                     'N':  100, 
                     'mu1': 0, 
                     'mu2': 0,
                     'sigma1': 0.01,
                     'sigma2': 0.01, 
                     'alpha': 0.9,
                     'lag': 1,
                     'seed': 10,
                     'expected_TE':0.2,
                     'expected_Z_score':0,
                     'tolerance':0.05
                    }
}
