from check_warnings import check_list
check_list()

import pandas as pd
import re

from nlp_stuff import void_term, validate_term, word_salad, non_empty_list

class Wordbag():
    def __init__(self):
        self.text = None
        self.includes = None
        self.body_only = None
        self.final_df = None
        self.path2label = {'': 404, 'neg': 0, 'only_pubis': 1, 'pos_pelvis': 2, 'femur': 4, 'ace_disloc': 5, 'complex': 777, 'si': 888, 'stable': 999}

    def prepare_criteria(self, includes_f='./includes_v2.xlsx', excludes_f='./excludes_v2.xlsx'):
        includes_excel = pd.read_excel(includes_f)
        femur_terms = includes_excel['femur'].dropna().tolist()
        pubis_terms = includes_excel['pubis'].dropna().tolist()
        si_terms = includes_excel['si'].dropna().tolist()
        acetabular_terms = includes_excel['acetabular'].dropna().tolist()
        dislocation_terms = includes_excel['dislocation'].dropna().tolist()
        stable_terms = includes_excel['stable'].dropna().tolist()
        general_terms = includes_excel['general'].dropna().tolist()
        ace_void = includes_excel['ace_void'].dropna().tolist()
        includes = femur_terms + pubis_terms + si_terms + acetabular_terms + dislocation_terms + stable_terms
        includes.sort()

        excludes_excel = pd.read_excel(excludes_f)

        inappropriate = excludes_excel['inappropriate'].dropna().tolist()

        hardware = excludes_excel['hardware'].dropna().tolist()
        treated = excludes_excel['treated'].dropna().tolist()
        equivocal = excludes_excel['equivocal'].dropna().tolist()
        uncertain = excludes_excel['uncertain'].dropna().tolist()
        soft_exclusions = treated + equivocal + uncertain

        nonacute = excludes_excel['old'].dropna().tolist()
        nonfracture = excludes_excel['nonfracture'].dropna().tolist()
        nontarget = excludes_excel['nontarget'].dropna().tolist()
        normals = excludes_excel['normal'].dropna().tolist()
        ignore_sentence_words = nonacute + nonfracture + nontarget + normals

        problem_terms = excludes_excel['problems'].dropna().tolist()

        self.femur_terms = femur_terms
        self.pubis_terms = pubis_terms
        self.si_terms = si_terms
        self.acetabular_terms = acetabular_terms
        self.ace_void = ace_void
        self.dislocation_terms = dislocation_terms
        self.stable_terms = stable_terms
        self.general_terms = general_terms
        self.paramount_terms = includes + general_terms

        self.hardware = hardware

        self.includes = includes
        self.soft_exclusions = soft_exclusions
        self.inappropriate = inappropriate
        self.ignore_sentence_words = ignore_sentence_words
        self.problem_terms = problem_terms

        a = pd.DataFrame({'femur': self.femur_terms})
        b = pd.DataFrame({'pubis': self.pubis_terms})
        c = pd.DataFrame({'si': self.si_terms})
        d = pd.DataFrame({'acetabular': self.acetabular_terms})
        e = pd.DataFrame({'dislocation': self.dislocation_terms})
        f = pd.DataFrame({'stable': self.stable_terms})
        g = pd.DataFrame({'general': self.general_terms})

        self.includes_df = pd.concat([a, b, c, d, e, f, g], axis=1)

    def clean_text(self, excel_file='./pelvis_fx_body.xlsx', imp_or_report_text='Report Text', inclusion_list=None,
                   post_inclusion_validate_list=None, alias_name='ID', item_divided=False, sentence_split=True,
                   increment_index=False, get_findings=False, use_inclusion_regex=False):

        raw_excel = pd.read_excel(excel_file)
        if increment_index:
            raw_excel.index = raw_excel.index + 1

        if item_divided: #if impression is split/divided by researcher, Item is case ID
            raw_excel = raw_excel.groupby([alias_name], sort=False, as_index=False).first()
            if increment_index:
                raw_excel.index = raw_excel.index + 1 #groupby will reindex from 0
            print('Check indexing since groupby will change the index values (starts from 0)')

        if imp_or_report_text in ['Impression', 'Report Text']:
            item_text = raw_excel[[alias_name, imp_or_report_text]].copy() # avoid setting on slice
            self.raw_text = item_text.copy()
            item_text[imp_or_report_text] = item_text[imp_or_report_text].str.lower()
            if imp_or_report_text == 'Report Text': # extract text after Findings heading
                addendum_or_mingle = re.compile(r'do not use|addendum')
                item_text['mingle_addendum'] = item_text[imp_or_report_text].map(lambda x: True if addendum_or_mingle.search(x) else False)

                if get_findings:
                    findings_rg = re.compile(r'findings(.*)', re.I | re.S)
                    item_text[imp_or_report_text] = item_text[imp_or_report_text].map(lambda x: x if findings_rg.search(x) else 'findingsNone') # put in findings if there are None.
                    item_text[imp_or_report_text] = item_text[imp_or_report_text].map(lambda x: findings_rg.search(x).group(1)) # Then just extract after the findings

                else: #get impression instead
                    impression_rg = re.compile(r'impression(.*)', re.I | re.S)
                    item_text[imp_or_report_text] = item_text[imp_or_report_text].map(lambda x: x if impression_rg.search(x) else 'impressionNone')  # put in findings if there are None.
                    item_text[imp_or_report_text] = item_text[imp_or_report_text].map(lambda x: impression_rg.search(x).group(1))  # Then just extract after the findings

        else:
            raise Exception("Select Impression or Report Text")

        self.item_text = item_text

        #if you also want to split by \n, maybe replace \n with '.' with regex before split

        split_imp = item_text.loc[:, imp_or_report_text].fillna('None').str.split('.', expand=False) #Impression now a list
        # pandas series str.split ignores nan, which is present when impression is empty, but using fillna, empty impression will now be None.

        if not sentence_split:
            split_imp = split_imp.map(lambda x: ['.'.join(x)])

        if inclusion_list is not None: #probably the same as the regex way
            dirty_frac_or_disloc = split_imp.map(lambda x: validate_term(x, word_list=inclusion_list))
        elif use_inclusion_regex: # the way I did it, regex way
            inclusion_criteria = re.compile('fracture|disloc|widen|diastasis', re.I)
            dirty_frac_or_disloc = split_imp.map(lambda x: [i for i in x if inclusion_criteria.search(i)])
        else:
            dirty_frac_or_disloc = split_imp

        clean_searched = dirty_frac_or_disloc.map(lambda x: [re.sub(r'\n', '', i) for i in x])
        stripped_searched = clean_searched.map(lambda x: [i.strip() for i in x])

        if post_inclusion_validate_list is not None: #use to encompass specific body parts ('rad', 'uln', etc.)
            stripped_searched = stripped_searched.map(lambda x: validate_term(x, post_inclusion_validate_list))

        text_df = pd.DataFrame(stripped_searched)
        text_df[alias_name] = item_text[alias_name]

        text_df['report_len'] = text_df['Report Text'].map(lambda x: len(x))
        text_df['report_bool'] = (text_df['report_len'] != 0)
        text_df['mingle_addendum'] = item_text['mingle_addendum']

        if self.text is None:
            print('first clean text method invocation')
            self.text = stripped_searched
            self.text_df = text_df
        else:
            print('second on clean text method invocation')
            self.second_pass = text_df

    def load_body_only(self, excel_file='./body_text_only.xlsx'):
        self.body_only = pd.read_excel(excel_file)

    def second_pass_hardware_removal(self):
        if self.final_df is None:
            raise Exception("Cannot do second pass without first doing first pass and getting a final_df to concat")
        expanded_includes_list = self.includes + self.general_terms
        exclusion_list = self.hardware
        self.clean_text(inclusion_list=exclusion_list, post_inclusion_validate_list=expanded_includes_list, get_findings=True)
        self.second_pass.rename(columns={'report_bool': 'potential_hardware', 'Report Text': 'hardware_text'}, inplace=True)
        merged_df = pd.concat([self.final_df, self.second_pass[['hardware_text', 'potential_hardware']]], axis=1)

        # looking for case SENSITIVE words from findings/FINDINGS
        case_sensitive_excludes = ['THA']
        raw_text = self.raw_text.copy()
        findings_rg = re.compile(r'(?:findings|FINDINGS)(.*)', re.I | re.S)
        raw_text['Report Text'] = raw_text['Report Text'].map(lambda x: x if findings_rg.search(x) else 'findingsNone')  # put in findings if there are None.
        raw_text['Report Text'] = raw_text['Report Text'].map(lambda x: findings_rg.search(x).group(1))
        split_imp = raw_text.loc[:, 'Report Text'].fillna('None').str.split('.', expand=False)
        dirty_frac_or_disloc = split_imp.map(lambda x: validate_term(x, word_list=case_sensitive_excludes))
        clean_searched = dirty_frac_or_disloc.map(lambda x: [re.sub(r'\n', '', i) for i in x])
        stripped_searched = clean_searched.map(lambda x: [i.strip() for i in x])
        stripped_searched = stripped_searched.map(lambda x: validate_term(x, expanded_includes_list))
        text_df = pd.DataFrame(stripped_searched)
        text_df['report_len'] = text_df['Report Text'].map(lambda x: len(x))
        text_df['report_bool'] = (text_df['report_len'] != 0)
        text_df.rename(columns={'report_bool': 'case_sens_bool', 'Report Text': 'case_sens_text'}, inplace=True)
        thirdy_merged_df = pd.concat([merged_df, text_df[['case_sens_bool', 'case_sens_text']]], axis=1)


        thirdy_merged_df = thirdy_merged_df[['ID', 'Report Text', 'relevant', 'potential_hardware', 'hardware_text',
                                             'problem_bool', 'problem', 'case_sens_bool', 'case_sens_text', 'no_fracture', 'pubis',
                                             'stable', 'si', 'femur','ace', 'dislocation', 'sep_label', 'multi_label']]

        self.lastly_merged_df = thirdy_merged_df

    def dichotomize_impression(self, exclude_hardware=True):

        if self.text is None:
            self.clean_text()

        if self.includes is None:
            self.prepare_criteria()

        text_series = self.text

        inappropriate_terms = self.inappropriate
        includes_terms = self.includes
        soft_exclusions = self.soft_exclusions
        general_terms = self.general_terms
        paramount_terms = self.paramount_terms
        hardware_terms = self.hardware

        # Remember that void and validate terms return_bool will be True/False for the WHOLE LIST if even a single word is found
        filter_terms = inappropriate_terms
        filter_bool = text_series.map(lambda x: void_term(x, filter_terms, return_bool=True))
        safe_text_series = text_series[filter_bool]

        paramount_text = safe_text_series.map(lambda x: validate_term(x, paramount_terms))
        paramount_text = paramount_text[paramount_text.map(non_empty_list)]
        # Only keep lists with paramount terms, if not, cases are excluded

        if exclude_hardware:
            exclude_terms = soft_exclusions + hardware_terms
        else:
            exclude_terms = soft_exclusions
        non_exclude_bool = paramount_text.map(lambda x: void_term(x, exclude_terms, return_bool=True))
        non_excluded_text = paramount_text[non_exclude_bool]

        #Regex filter more excludes
        fx_rg = re.compile(r'(?:cannot).*(?:exclude|rule )', re.I | re.S)
        regex_exclude_bool = non_excluded_text.map(lambda x: [i for i in x if fx_rg.search(i)]).map(non_empty_list)
        # if match, then True and has the phrase, but cannot do the other way, since it'll capture sentences that don't match
        excludes_done_text = non_excluded_text[~regex_exclude_bool]

        post_suppresed_text = excludes_done_text.map(lambda x: void_term(x, self.ignore_sentence_words)) #if empty list, negative

        # Only place to explicitly suppress negation
        fx_rg = re.compile(r'(?:without|no |negative).*(?:fracture|malalignment)', re.I | re.S)
        regex_positives = post_suppresed_text.map(lambda x: [i for i in x if not fx_rg.search(i)])
        positive_bool = regex_positives.map(non_empty_list)

        prob_positive = regex_positives[positive_bool]
        negative_cases = post_suppresed_text[~positive_bool]

        self.pos_positive = prob_positive # subsequently non-positive cases remain EXCLUDED.  Generic with general terms included.
        self.negative_cases = negative_cases # NO change to negative cases beyond here.

    def split_positives(self):

        pos_positive, negative_cases = self.pos_positive, self.negative_cases
        femur_terms, pubis_terms, si_terms = self.femur_terms, self.pubis_terms, self.si_terms
        acetabular_terms, stable_terms, dislocation_terms = self.acetabular_terms, self.stable_terms, self.dislocation_terms

        joint_reduced_terms = ['located', 'reduc', 'congruent', 'relocation', 'prior', 'normal']
        incomplete_reduction_terms = ['partial ']

        # Potentially overlapping cases
        pubis_cases = pos_positive.map(lambda x: validate_term(x, pubis_terms))

        acetabular_cases = pos_positive\
            .map(lambda x: validate_term(x, acetabular_terms))\
            .map(lambda x: void_term(x, self.ace_void))

        femur_cases = pos_positive.map(lambda x: validate_term(x, femur_terms))
        si_cases = pos_positive.map(lambda x: validate_term(x, si_terms))
        stable_cases = pos_positive.map(lambda x: validate_term(x, stable_terms))

        dislocation_cases = pos_positive.map(lambda x: validate_term(x, dislocation_terms)) \
            .map(lambda x: void_term(x, joint_reduced_terms))
        disloc_rg = re.compile(r'(?:without|no ).*dislocation', re.I | re.S)
        dislocation_cases = dislocation_cases.map(lambda x: [i for i in x if not disloc_rg.search(i)])

        femur_bool = femur_cases.map(non_empty_list)
        pubis_bool = pubis_cases.map(non_empty_list)
        ace_bool = acetabular_cases.map(non_empty_list)
        si_bool = si_cases.map(non_empty_list)
        stable_bool = stable_cases.map(non_empty_list)
        dislocation_bool = dislocation_cases.map(non_empty_list)
        pos_pelvic_bool = si_bool | stable_bool
        ace_disloc_bool = ace_bool | dislocation_bool

        fem_index = femur_cases[femur_bool].index
        pubis_index = pubis_cases[pubis_bool].index
        ace_index = acetabular_cases[ace_bool].index
        si_index = si_cases[si_bool].index
        stable_index = stable_cases[stable_bool].index
        disloc_index = dislocation_cases[dislocation_bool].index

        problem_lines = pos_positive.map(lambda x: validate_term(x, self.problem_terms))
        problem_bool = problem_lines.map(non_empty_list)

        pos_pos_df = pd.DataFrame({'femur': femur_cases, 'femur_bool': femur_bool,
                               'pubis': pubis_cases, 'pubis_bool': pubis_bool,
                               'ace': acetabular_cases, 'ace_bool': ace_bool,
                               'si': si_cases, 'si_bool': si_bool,
                               'stable': stable_cases, 'stable_bool': stable_bool,
                               'dislocation': dislocation_cases, 'disloc_bool': dislocation_bool,
                               'problem_bool': problem_bool, 'problem': problem_lines,
                                'pos_pelvic_bool': pos_pelvic_bool, 'ace_disloc_bool': ace_disloc_bool
                               })

        true_pos_index = fem_index.union(pubis_index).union(ace_index).union(si_index).union(stable_index).union(disloc_index)

        #Checking exclusivity
        #Isolate classes from the true positives
        f_pos_cases = pos_pos_df.loc[true_pos_index].copy()

        femur_screen = (f_pos_cases['ace_bool'] | f_pos_cases['pubis_bool'] | f_pos_cases['disloc_bool'] |
                      f_pos_cases['si_bool'] | f_pos_cases['stable_bool']) == False
        
        ace_screen = (f_pos_cases['femur_bool'] | f_pos_cases['pubis_bool'] | f_pos_cases['disloc_bool'] |
                      f_pos_cases['si_bool'] | f_pos_cases['stable_bool']) == False
        
        pubis_screen = (f_pos_cases['ace_bool'] | f_pos_cases['femur_bool'] | f_pos_cases['disloc_bool'] |
                      f_pos_cases['si_bool'] | f_pos_cases['stable_bool']) == False
        
        dislocation_screen = (f_pos_cases['ace_bool'] | f_pos_cases['femur_bool'] | f_pos_cases['pubis_bool'] |
                              f_pos_cases['si_bool'] | f_pos_cases['stable_bool']) == False
        
        si_screen = (f_pos_cases['ace_bool'] | f_pos_cases['pubis_bool'] | f_pos_cases['disloc_bool'] |
                           f_pos_cases['femur_bool'] | f_pos_cases['stable_bool']) == False
        
        stable_screen = (f_pos_cases['ace_bool'] | f_pos_cases['pubis_bool'] | f_pos_cases['disloc_bool'] |
                           f_pos_cases['femur_bool'] | f_pos_cases['si_bool']) == False

        #For complex cases, all of the screens will end up as False since at least 2 parts bool will be True
        single_site_screen = femur_screen | ace_screen | pubis_screen | dislocation_screen | si_screen | stable_screen

        # combo cases
        ace_disloc_combo_screen = (f_pos_cases['femur_bool'] | f_pos_cases['pubis_bool'] | f_pos_cases['si_bool'] |
                                   f_pos_cases['stable_bool']) == False
        pos_pelvic_combo_screen = (f_pos_cases['ace_bool'] | f_pos_cases['pubis_bool'] | f_pos_cases['disloc_bool'] |
                                   f_pos_cases['femur_bool']) == False
        sep_class_screen = (femur_screen | pubis_screen | ace_disloc_combo_screen | pos_pelvic_combo_screen)

        pelvic_ring_screen = (f_pos_cases['ace_bool'] | f_pos_cases['disloc_bool'] | f_pos_cases['femur_bool']) == False
        consolid_class_screen = (femur_screen | ace_disloc_combo_screen | pelvic_ring_screen)

        #Exclusive indices
        only_femur_index = f_pos_cases[femur_screen].index
        only_ace_index = f_pos_cases[ace_screen].index
        only_pubis_index = f_pos_cases[pubis_screen].index
        only_dislocation_index = f_pos_cases[dislocation_screen].index
        only_si_index = f_pos_cases[si_screen].index
        only_stable_index = f_pos_cases[stable_screen].index
        ace_disloc_index = f_pos_cases[ace_disloc_combo_screen].index
        pos_pelvic_index = f_pos_cases[pos_pelvic_combo_screen].index
        pelvic_ring_index = f_pos_cases[pelvic_ring_screen].index

        # Decide what constitutes a complex cass by selecting screen
        # Pelvic ring fracture as 1 class
        consolid_complex_cases = f_pos_cases[~consolid_class_screen]
        consolid_complex_case_index = consolid_complex_cases.index

        # Pelvic ring fracture as anterior or posterior
        sep_complex_cases = f_pos_cases[~sep_class_screen]
        sep_complex_case_index = sep_complex_cases.index

        # Create bools for positive classes
        f_pos_cases['only_pubis'] = f_pos_cases.index.isin(only_pubis_index)

        f_pos_cases['only_si'] = f_pos_cases.index.isin(only_si_index)
        f_pos_cases['only_stable'] = f_pos_cases.index.isin(only_stable_index)
        f_pos_cases['pos_pelvis'] = f_pos_cases.index.isin(pos_pelvic_index)

        f_pos_cases['pelvic_ring'] = f_pos_cases.index.isin(pelvic_ring_index)

        f_pos_cases['only_femur'] = f_pos_cases.index.isin(only_femur_index)

        f_pos_cases['only_ace'] = f_pos_cases.index.isin(only_ace_index)
        f_pos_cases['only_dislocation'] = f_pos_cases.index.isin(only_dislocation_index)
        f_pos_cases['ace_disloc'] = f_pos_cases.index.isin(ace_disloc_index)

        f_pos_cases['consolid_complex_cases'] = f_pos_cases.index.isin(consolid_complex_case_index)
        f_pos_cases['sep_complex_cases'] = f_pos_cases.index.isin(sep_complex_case_index)

        self.pos_case_df = f_pos_cases

    def labeling(self):

        neg_series = self.negative_cases
        neg_df = pd.DataFrame(neg_series)
        neg_df['consold_label'] = 0
        neg_df['sep_label'] = 0
        neg_df['problem'] = neg_df['Report Text'].map(lambda x: validate_term(x, self.problem_terms))
        neg_df['problem_bool'] = neg_df['problem'].map(non_empty_list)
        neg_df['multi_label'] = 'neg'
        neg_df['multi_label'] = neg_df['multi_label'].map(lambda x: [self.path2label[i] for i in x.strip().split(' ')])

        pos_df = self.pos_case_df

        a = pos_df['only_pubis'] == True
        b = pos_df['pos_pelvis'] == True
        c = pos_df['pelvic_ring'] == True

        d = pos_df['only_femur'] == True

        e = pos_df['ace_disloc'] == True

        f1 = pos_df['consolid_complex_cases'] == True
        f2 = pos_df['sep_complex_cases'] == True

        # Label the cases based on index of created filters
        pos_df['sep_label'] = 404
        pos_df.loc[pos_df[a].index, ['sep_label']] = 1
        pos_df.loc[pos_df[b].index, ['sep_label']] = 2
        # pos_df.loc[pos_df[c].index, ['sep_label']] = 3
        pos_df.loc[pos_df[d].index, ['sep_label']] = 4
        pos_df.loc[pos_df[e].index, ['sep_label']] = 5
        pos_df.loc[pos_df[f2].index, ['sep_label']] = 6

        pos_df['consold_label'] = 404
        # pos_df.loc[pos_df[a].index, ['consold_label']] = 1
        # pos_df.loc[pos_df[b].index, ['consold_label']] = 2
        pos_df.loc[pos_df[c].index, ['consold_label']] = 3
        pos_df.loc[pos_df[d].index, ['consold_label']] = 4
        pos_df.loc[pos_df[e].index, ['consold_label']] = 5
        pos_df.loc[pos_df[f1].index, ['consold_label']] = 6

        pos_df['multi_label'] = ''
        pos_df.loc[pos_df[pos_df['pubis_bool']].index, ['multi_label']] = pos_df.loc[pos_df[pos_df['pubis_bool']].index, ['multi_label']].applymap(lambda x: x + 'only_pubis ')
        pos_df.loc[pos_df[pos_df['pos_pelvic_bool']].index, ['multi_label']] = pos_df.loc[pos_df[pos_df['pos_pelvic_bool']].index, ['multi_label']].applymap(lambda x: x + 'pos_pelvis ')
        pos_df.loc[pos_df[pos_df['femur_bool']].index, ['multi_label']] = pos_df.loc[pos_df[pos_df['femur_bool']].index, ['multi_label']].applymap(lambda x: x + 'femur ')
        pos_df.loc[pos_df[pos_df['ace_disloc_bool']].index, ['multi_label']] = pos_df.loc[pos_df[pos_df['ace_disloc_bool']].index, ['multi_label']].applymap(lambda x: x + 'ace_disloc ')

        pos_df['multi_label'] = pos_df['multi_label'].map(lambda x: [self.path2label[i] for i in x.strip().split(' ')])

        pos_df = pos_df[['pubis', 'stable', 'si', 'femur', 'ace', 'dislocation', 'sep_label', 'multi_label', 'problem', 'problem_bool']]

        self.final_pos_df = pos_df
        self.final_neg_df = neg_df

        #The way to put the pos and negative cases back into the whole excel
        whole_df = self.text_df.copy()
        irrelevant_idx = whole_df.index.difference(pos_df.index).difference(neg_df.index)
        whole_df['relevant'] = True
        whole_df.loc[irrelevant_idx, 'relevant'] = False
        mingle_idx = whole_df[whole_df['mingle_addendum']].index
        whole_df.loc[mingle_idx, 'relevant'] = False

        whole_df = whole_df[['Report Text', 'ID', 'relevant']]
        whole_df[['pubis', 'stable', 'si', 'femur', 'ace', 'dislocation', 'sep_label', 'multi_label', 'problem', 'problem_bool']] = \
            pos_df[['pubis', 'stable', 'si', 'femur', 'ace', 'dislocation', 'sep_label', 'multi_label', 'problem', 'problem_bool']]
        whole_df.loc[neg_df.index, ['sep_label', 'multi_label', 'problem', 'problem_bool']] = neg_df[['sep_label', 'multi_label', 'problem', 'problem_bool']]
        whole_df['no_fracture'] = neg_df['Report Text']
        whole_df = whole_df.fillna('impertinent')

        self.final_df = whole_df

    def do_everything(self):
        self.clean_text()
        self.dichotomize_impression()
        self.split_positives()
        self.labeling()
        self.second_pass_hardware_removal()

    def do_first_pass(self):
        self.clean_text()
        self.dichotomize_impression()
        self.split_positives()
        self.labeling()