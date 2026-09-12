from pathlib import Path
p=Path('/root/autodl-tmp/aic_race/D-FINE/src/solver/_solver.py')
s=p.read_text();old='''        ]

    def _setup(self):''';new='''        ]
        # AIC 12-class custom mapping into Objects365 classifier rows.
        self.obj365_ids = [0, 21, 92, 2, 89, 46, 5, 156, 40, 44, 114, 183]

    def _setup(self):'''
if old not in s: raise SystemExit('anchor missing')
p.write_text(s.replace(old,new,1))
print('PATCHED', 'self.obj365_ids = [0, 21, 92' in p.read_text())
