
class util:
    @staticmethod
    def progress_meter(progress, done_sign='|', remain_sign=' ', bar_len=30):
        done = int(progress * bar_len)
        remain = bar_len - done
        msg = f"|{done*done_sign}{remain*remain_sign}| : {round(100*progress, 2)}%"
        return msg
    
    @staticmethod
    def nice_print(msg, width=80, align='center', startswith='>>', endswith=''):
        space = ' '
        used_space = int(len(startswith) + len(endswith) + len(msg))
        if align == 'left':
            left_space = 2 * space
            len_right_space = int(width - used_space - len(left_space))
            right_space = space * len_right_space
        elif align == 'center':
            len_left_space = int(.5 * (width - used_space))
            len_right_space = int(width - len_left_space - used_space)
            left_space = space * len_left_space
            right_space = space * len_right_space
        toprint = f'{startswith}{left_space}{msg}{right_space}{endswith}'
        print(f"{toprint:^80}")
    
    @staticmethod
    def print_dict(dicts, startswith='', sort_value=True):
        if sort_value:
            lines = sorted([(value, key) for value,key in zip(dicts.values(), dicts.keys())], reverse=True)
        else:
            lines = [(value,key) for key,value in sorted(dicts.items())]
        # the objective is to print dict as this format: key : value
        len_max = max(list(map(len, dicts.keys()))) + len(startswith)
        space = ' '
        util.nice_print(msg=startswith, align='left')
        for line in lines:
            left = space * len(startswith)
            blank = int(len_max - len(line[1]) - len(left))
            msg = f"{left} {line[1]}{space * blank} : {line[0]}"
            util.nice_print(msg, align='left')

    @staticmethod
    def print_list(nlist, startswith=' ', style='default'):
        space = ' '
        if style == 'default':
            pass
        elif style == 'lower':
            nlist = [i.lower() for i in nlist]
        elif style == 'upper':
            nlist = [i.upper() for i in nlist]

        util.nice_print(msg=startswith, align='left')
        for _ in nlist:
            util.nice_print(f'{space*len(startswith)} {_}', align='left', endswith='')

    @staticmethod
    def to_latex_table(data, fname):
        for _ in data.columns:
            try:
                data[_] = data[_].astype(float)
            except:
                pass
        data.to_latex(fname, float_format="{:,.2f}".format,
                      longtable=True, index=False, na_rep='-')
        
        with open(fname, 'r') as f:
            tex = f.readlines()
        slash = "\\"
        kb = "{"
        kt = "}"
        eol = " \\\\\n"
        line = tex[2].strip().replace("\\","").split(' & ')
        new_line = ' & '.join([f"{slash}multicolumn{kb}1{kt}{kb}c{kt}{kb}{i.strip().title()}{kt}" for i in line])
        new_line = new_line + eol
        tex[2] = new_line
        tex[6] = new_line
        tex[10] = tex[10].replace("Continued on next page", "Continued..")
        with open(fname, 'w') as f:
            f.writelines(tex)