# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:48:01 2020

@author: e-sshen
"""
import pandas as pd
import numpy as np
import datetime
import time
import math
import xlrd
from datetime import date,timedelta

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, type(dict)):
         pretty(value, indent+1)
      elif isinstance(value,type(list)):
          for i in value:
              print('\t' * (indent+1) + str(i))
      else:    
         print('\t' * (indent+1) + str(value))
         
def is_nan(x):
    return (x is np.nan or x != x)

class Transaction:
    def __init__ (self, tr_id, date, activity, days, tr_amount):
        self.tr_id = tr_id
        self.date = date
        self.activity = activity
        self.days = days
        self.tr_amount = tr_amount
        
    def print(self):
        print('\t\t' + str(self.tr_id))
        print('\t\t' + str(self.date))
        print('\t\t' + str(self.activity))
        print('\t\t' + str(self.days))
        print('\t\t' + str(self.tr_amount))
        
    def week(self):
        return (self.date).isocalendar()[1]

class Account:   
    def __init__(self, account, term, interest, productType, contract, termstart, termend, balance = None, transactions = None, ):
        self.account = account
        if transactions is None:
            transactions = []
        self.transactions = transactions
        self.term = term
        self.interest = interest
        self.productType = productType
        self.contract = contract
        self.termstart = termstart
        self.termend = termend
        if balance is None:
            balance = {}
        self.balance = balance
    
    def print(self):
        print('\t' + str(self.account))
        print('\t' + str(self.term))
        print('\t' + str(self.interest))
        print('\t' + str(self.productType))
        print('\t' + str(self.contract))
        print('\t' + str(self.termstart))
        print('\t' + str(self.termend))
        pretty(self.balance)
        for i in self.transactions:
            i.print()
            
    def sav_flag(self):
        return self.productType == 'SAV'
    def chk_flag(self):
        return self.productType == 'CHK'
    def gic_flag(self):
        return self.productType == 'GIC'
    def reg_flag(self):
        reg = ['CSH3', 'CSH9', 'RRIF', 'RRSP', 'STD', 'TFSA']
        if self.productType in reg:
            return True
        else:
            return False
    def most_recent(self, week):
        if self.balance is not None:
            bal = []
            for i in self.balance:
                if i.isocalendar()[1] >= week:
                    bal.append(i)
            if bal != []:
                balance = sorted(bal, reverse = True)[0]
            else:
                balance = None
        else:
            balance = None
        
        transacc = []
        for i in self.transactions:
            if i.date.isocalendar()[1] >= week:
                transacc.append(i.date)
        if len(transacc) == 0:
            transaction = None
        else:
            transaction = sorted(transacc, reverse = True)[0]
        
        return balance, transaction
            
        
    def calc_term(self):
        if self.reg_flag() or self.gic_flag():
            start = self.termstart
            end = self.termend
            return (end - start).days
    
    def active_flag(self, week):
        if self.reg_flag() or self.gic_flag():
            if ((self.termend).isocalendar()[1] + 12) >= 53:
                return True
            if ((self.termend).year) > 2019:
                return True
            if ((self.termend).isocalendar()[1] + 12 )>= week:
                return True
            return False
        else:
            bal, tr = self.most_recent(week)
            if tr is None and bal is None:
                return False
            if tr is None:
                if bal >= datetime.date(2019,12,1):
                    return True
                if bal.isocalendar()[1] + 12 >= 53:
                    return True
                if bal.year > 2019:
                    return True
                if (bal.isocalendar()[1] + 12) >= week:
                    return True
            if bal is None:
                if tr >= datetime.date(2019,12,1):
                    return True
                if tr.isocalendar()[1] + 12 >= 53:
                    return True
                if tr.year > 2019:
                    return True
                if (tr.isocalendar()[1] + 12) >= week:
                    return True
            else:
                if bal >= datetime.date(2019,12,1) or tr >= datetime.date(2019,12,1):
                    return True
                if bal.isocalendar()[1] + 12 >= 53 or tr.isocalendar()[1] +12 >= 53:
                    return True
                if bal.year > 2019 or tr.year > 2019:
                    return True
                if (bal.isocalendar()[1] + 12) >= week or (tr.isocalendar()[1] + 12) >= week:
                    return True
            return False
        
    def gic_purchase(self, week):
        for tr in self.transactions:
            if tr.week() == week:
                if tr.activity == 'New Issue           ':
                    return True
        return False
    def gic_renewal(self, week):
        for tr in self.transactions:
            if tr.week() == week:
                if tr.activity == 'Renewed In          ':
                    return True
        return False
    
    def balance_lookup(self, week):
        balances = [(k,v) for k,v in (self.balance).items()]        
        if self.sav_flag():
            for i in (balances):
                if i[0].isocalendar()[1] >= week:
                    return i[1]
            return balances[-1][1]
    
    def average_balance(self, week):
        if self.sav_flag():
            sum = 0
            count = 0
            for i in range(0,53):
                if i >= week:
                    sum = sum + float(self.balance_lookup(i).replace(',', ''))
                    count = count + 1
            if count == 0:
                return 0
            else:
                return sum/count
        else:
            if self.contract is None:
                return 0
            else:
                return self.contract
    
    def balance_volatility(self, week):
        if self.sav_flag():
            vol = []
            for i in range(0,53):
                if i >= week:
                    elem = float(self.balance_lookup(i).replace(',','')) - self.average_balance(week)
                    elem = math.sqrt(abs(elem))
                    vol.append(elem)
            sum = 0
            for i in vol:
                sum = sum + i
            if len(vol) == 0:
                return 0
            else:
                return sum/len(vol)
        else:
            return 0
    def time_to_balance(self, week):
        balances = [(k,v) for k,v in (self.balance).items()]        
        if self.sav_flag():
            sum = 0
            count = 0
            for i in range(1, len(balances)):
                if balances[i][0].isocalendar()[1] >= week:
                    delta = (balances[i][0] - balances[i-1][0]).days
                    sum = sum + delta
                    count = count + 1
            if count == 0:
                return 0
            else:
                return sum/count
        else:
            if self.calc_term() is None:
                return 0
            else:
                return self.calc_term()
            
        
    def trans_average(self, week):            
        sum = 0
        count = 0
        for i in self.transactions:
            if i.tr_amount is not None and i.tr_amount != 'None':
                if i.date.isocalendar()[1] >= week:
                    sum = sum + float(i.tr_amount.replace(',',''))
                    count = count + 1
        if self.reg_flag() or self.sav_flag():
            if self.contract is None:
                return 0
            else:
                return self.contract
        if count == 0:
            return 0
        else:
            return sum/count
        
    def trans_volatility(self, week):
        vol = []
        for i in self.transactions:
            if i.tr_amount is not None and i.tr_amount != 'None':
                if i.date.isocalendar()[1] >= week:
                    elem = float(str(i.tr_amount).replace(',','')) - self.trans_average(week)
                    elem = math.sqrt(abs(elem))
                    vol.append(elem)
        sum = 0
        for i in vol:
            sum = sum + i
        if len(vol) == 0:
            return 0
        else:
            return sum/len(vol)
        
    def num_trans(self, week):
        count = 0
        for i in self.transactions:
            if i.date.isocalendar()[1] >= week:
                count = count + 1
        return count
            
        
    def time_to_trans(self, week):       
        sum = 0
        count = 0
        for i in range(1, len(self.transactions)):
            if (self.transactions)[i].date.isocalendar()[1] >= week:
                delta = ((self.transactions)[i].date - (self.transactions)[i-1].date).days
                sum = sum + delta
                count = count + 1
        if self.sav_flag() or self.reg_flag():
            if self.calc_term() is None:
                return 0
            else:
                return self.calc_term()
        if count == 0:
            return 0
        else:
            return sum/count    
    
class Holder:
    accounts = []
    
    def __init__(self, holder, birthdate, addeddate, type, firstname, lastname, gender, organization, fsa, channel, accounts = None):
        self.holder = holder
        if accounts is None:
            accounts = []
        self.accounts = accounts
        self.birthdate = birthdate
        self.addeddate = addeddate
        self.type = type
        self.firstname = firstname
        self.lastname = lastname
        self.gender = gender
        self.organization = organization
        self.fsa = fsa
        self.channel = channel

        
    def print(self):
        print(self.holder)
        print(self.birthdate)
        print(self.addeddate)
        print(self.type)
        print(self.firstname)
        print(self.lastname)
        print(self.gender)
        print(self.organization)
        print(self.fsa)
        print(self.channel)
        for i in self.accounts:
            i.print()
            
    def average(lt):
        sum = 0
        count = 0
        for i in lt:
            sum = sum + i
            count = count + 1
        if count == 0:
            return 0
        return sum/count
            
    def active_acc(self, week):
        count = 0
        for i in self.accounts:
            if i.active_flag(week):
                count = count + 1
        return count
    def num_acc(self):
        return len(self.accounts)
    def age(self, week):
        start = datetime.datetime(2019,1,2)
        if self.birthdate is None:
            return None
        else:
            delta = timedelta(weeks = week)
            curr = start + delta
            return (curr.date() - self.birthdate)
    def customer_age(self, week):
        start = datetime.datetime(2019,1,2)
        if self.addeddate is None:
            return None
        else:
            delta = timedelta(weeks = week)
            curr = start + delta
            return (curr.date() - self.addeddate)
    def total_product_vector(self):
        products = {'CHK':0,
                    'CSH3':0, 
                    'CSH9':0, 
                    'GIC':0, 
                    'RRIF':0,
                    'RRSP':0,
                    'RSPS':0,
                    'SAV':0,
                    'STD':0,
                    'TFSA':0}
        for i in self.accounts:
            products[i.productType] = products[i.productType] + 1
        return products
    def active_product_vector(self, week):
        products = {'CHK':0,
                    'CSH3':0, 
                    'CSH9':0, 
                    'GIC':0, 
                    'RRIF':0,
                    'RRSP':0,
                    'RSPS':0,
                    'SAV':0,
                    'STD':0,
                    'TFSA':0}
        for i in self.accounts:
            if i.active_flag(week):
                products[i.productType] = products[i.productType] + 1
        return products
    
    def average_contract(self, week):
        sum = 0
        count = 0
        for i in self.accounts:
            if i.reg_flag() or i.gic_flag():
                if i.termstart.isocalendar()[1] >= week:
                    sum = sum + i.contract
                    count = count + 1
        if count == 0:
            return 0
        else:
            return sum/count
    def average_interest(self, week):
        sum = 0
        count = 0
        for i in self.accounts:
            if i.reg_flag() or i.gic_flag():
                if i.termstart.isocalendar()[1] >= week:
                    sum = sum + i.interest
                    count = count + 1
        if count == 0:
            return 0
        else:
            return sum/count
    def average_term(self, week):
        sum = 0
        count = 0
        for i in self.accounts:
            if i.reg_flag() or i.gic_flag():
                if i.termstart.isocalendar()[1] >= week:
                    sum = sum + i.calc_term()
                    count = count + 1
        if count == 0:
            return 0
        else:
            return sum/count
    def active_flag(self, week):
        for i in self.accounts:
            if i.active_flag(week):
                return True
        return False
    
    def gic_purchase(self, week):
        for i in self.accounts:
            if i.gic_purchase(week):
                return True
        return False
    def gic_renewal(self, week):
        for i in self.accounts:
            if i.gic_renewal(week):
                return True
        return False
    
    def gic_sav_order(self):
        status = 0
        first = 'Other'
        for i in self.accounts:
            if i.productType == 'GIC' and status == 0:
                status = 1
                first = 'GIC'
            elif i.productType == 'SAV' and status == 1:
                status = 3
                first = 'GIC'
            elif i.productType == 'GIC' and status == 2: 
                status = 3
                first = 'SAV'
            elif i.productType == 'SAV' and status == 0:    
                status = 2
                first = 'SAV'
        return status, first

    
    def sav_char(self, week):
        multiple = False
        status, first = self.gic_sav_order()
        if status == 3 or status == 2:
            for i in self.accounts:
                if i.productType == 'SAV':
                    multiple = True
                    return multiple, i.balance_lookup(week), i.average_balance(week), i.balance_volatility(week), i.time_to_balance(week)
        else:
            lookup_sum = 0
            lookup_count = 0
            avg_sum = 0
            avg_count = 0
            vol_sum = 0
            vol_count = 0
            time_sum = 0
            time_count = 0            
            for i in self.accounts:
                if i.productType == 'SAV':
                    lookup_sum = lookup_sum + (i.balance_lookup(week))
                    lookup_count = lookup_count + 1
                    avg_sum = avg_sum + i.average_balance(week)
                    avg_count = avg_count + 1
                    vol_sum = vol_sum + i.balance_volatility(week)
                    vol_count = vol_count + 1
                    time_sum = time_sum + i.time_to_balance(week)
                    time_count = time_count + 1
            if lookup_count == 0:
                lookup_avg = 0
            else:
                lookup_avg = lookup_sum / lookup_count
            if avg_count == 0:
                avg_avg = 0
            else:
                avg_avg = avg_sum / avg_count
            if vol_count == 0:
               vol_avg = 0
            else:
                vol_avg = vol_sum / vol_count
            if time_count == 0:
                time_avg = 0
            else:
                time_avg = time_sum / time_count
        return multiple, lookup_avg, avg_avg, vol_avg, time_avg
    
    def gic_char(self, week):
        num_sum = 0
        num_count = 0
        avg_sum = 0
        avg_count = 0
        vol_sum = 0
        vol_count = 0
        time_sum = 0
        time_count = 0
        for i in self.accounts:
            if i.reg_flag() or i.gic_flag() and i.active_flag(week):
                num_sum = num_sum + (i.num_trans(week))
                num_count = num_count + 1
                avg_sum = avg_sum + i.trans_average(week)
                avg_count = avg_count + 1
                vol_sum = vol_sum + i.trans_volatility(week)
                vol_count = vol_count + 1
                time_sum = time_sum + i.time_to_trans(week)
                time_count = time_count + 1
        if num_count == 0:
            num_avg = 0
        else:
            num_avg = num_sum / num_count
        if avg_count == 0:
            avg_avg = 0
        else:
            avg_avg = avg_sum / avg_count
        if vol_count == 0:
            vol_avg = 0
        else:
            vol_avg = vol_sum / vol_count
        if time_count == 0:
            time_avg = 0
        else:
            time_avg = time_sum / time_count
        return num_avg, avg_avg, vol_avg, time_avg
        
         

def aggregate(dict):
    def create_transaction(key, dict):
        tr_id = key
        date = dict[key]['Date']
        activity = ''
        if 'Activity' in dict[key]:
            activity = dict[key]['Activity']
            
        days = 0
        if 'Days to Maturity' in dict[key]:
            days = dict[key]['Days to Maturity']
        else:
            days = None
        
        tr_amount = 0
        if 'Transaction Amount' in dict[key]:
            tr_amount = dict[key]['Transaction Amount']
        else:
            tr_amount = None
        
        new_transaction = Transaction(tr_id, date, activity, days, tr_amount)
        return new_transaction
    
    def create_account(key, dict):
        transactions = []
        term = dict[key]['Term']
        interest = dict[key]['Interest']
        productType = dict[key]['Product Type']
        if productType == 'SAV':
            balance = dict[key]['Balance']
        else:
            balance = {}
        contract = dict[key]['Contract Principal']
        termstart = dict[key]['Term Start']
        termend = dict[key]['Term End']
        for x in dict[key]['Transactions']:
            if x not in transactions:
                transactions.append(create_transaction(x, dict[key]['Transactions']))        
        account = key
        new_account = Account(account,term, interest, productType, contract, termstart, termend, balance, transactions)
        return new_account
    
    def create_holder(key, dict):
        accounts = []
        birthdate = dict[key]['Date of Birth']
        addeddate = dict[key]['Date Added']
        type = dict[key]['Type']
        firstname = dict[key]['First Name']
        lastname = dict[key]['Last Name']
        gender = dict[key]['Gender']
        organization = dict[key]['Organization']
        fsa = dict[key]['FSA']
        channel = dict[key]['Channel']
        for x in dict[key]['Accounts']:
            if x not in accounts:
                accounts.append(create_account(x, dict[key]['Accounts']))
        
        holder = key
        new_holder = Holder(holder, birthdate, addeddate, type, firstname, lastname, gender, organization, fsa, channel, accounts,)
        
        return new_holder
    
    holders = []
  
    for i in dict:
        holders.append(create_holder(i, dict))
    print('Creating holders')
    return holders
    
         

def populatedict(data_list, holder_list, sav_list, week): 
    def transaction_details(data_list, i):
        date = pd.to_datetime(data_list[i][2]).date()
        sav_activity = data_list[i][7]
        sav_amount = data_list[i][8]    
        activity = data_list[i][9]
        maturity = data_list[i][15]
    
        if is_nan(activity):
            details= {'Date': date,
                      'Activity': sav_activity,
                      'Transaction Amount': sav_amount}
        else:
            details = {'Date': date,
                       'Activity': activity,
                       'Days to Maturity': maturity}
        return details
           
    def account_details(data_list, i):
        term = data_list[i][3]
        interest = data_list[i][4]
        product = data_list[i][6]
        contract = data_list[i][11]
        termstart = pd.to_datetime(data_list[i][12]).date()
        termend = pd.to_datetime(data_list[i][13]).date()
    
        details =  {'Term': term,
                    'Interest': interest,
                    'Product Type': product,
                    'Contract Principal': contract,
                    'Term Start': termstart,
                    'Term End': termend}
        return details
    
    def populateholder(holder_list):
        holder_dict = {}
    
        for i in range(0,len(holder_list)):
            holder = holder_list[i][0]        
        
            if holder not in holder_dict:
                holder_dict[holder] = {'Date of Birth': holder_list[i][1],
                                       'Date Added': holder_list[i][2],
                                       'Type': holder_list[i][3],
                                       'First Name': holder_list[i][4],
                                       'Last Name': holder_list[i][5],
                                       'Gender': holder_list[i][6],
                                       'Organization': holder_list[i][7],
                                       'FSA': holder_list[i][8]}
        print('Populating holder information')
        return holder_dict
    
    def combineholders(dict, holder_dict):
        for holder in dict:
            if holder in holder_dict:
                    dict[holder]['Date of Birth']= pd.to_datetime(holder_dict[holder]['Date of Birth']).date()
                    dict[holder]['Date Added']= pd.to_datetime(holder_dict[holder]['Date Added']).date()
                    dict[holder]['Type']= holder_dict[holder]['Type']
                    dict[holder]['First Name']= holder_dict[holder]['First Name']
                    dict[holder]['Last Name']= holder_dict[holder]['Last Name']
                    dict[holder]['Gender']= holder_dict[holder]['Gender']
                    dict[holder]['Organization']= holder_dict[holder]['Organization']
                    dict[holder]['FSA']= holder_dict[holder]['FSA']
            else:
                    dict[holder]['Date of Birth']= datetime.date(1900,1,1)#CHANGE
                    dict[holder]['Date Added']= datetime.date(1900,1,1)#CHANGE
                    dict[holder]['Type']= "None"
                    dict[holder]['First Name']= "None"
                    dict[holder]['Last Name']= "None"
                    dict[holder]['Gender']= "None"
                    dict[holder]['Organization']= "None"
                    dict[holder]['FSA']= "M1T" #CHANGE TO MORE SUITABLE DEFAULT        
        print('Combining holder information')
        return dict
    
    def populatebalance(sav_list):
        balance_dict = {}
        for i in sav_list:
            holder = i[0]
            account = i[1]
            Date = pd.to_datetime(i[2]).date()
            balance = i[3]
        
            if week >= Date.isocalendar()[1]:
                if holder in balance_dict:
                    balance_dict[holder]['Account'] = account
                    if Date not in balance_dict[holder]['Balance']:    
                        balance_dict[holder]['Balance'][Date] = balance
                else:
                    balance_dict[holder] = {'Account': account,
                                        'Balance': {Date: balance}}
        
        print('Populating balance information')
        return balance_dict
    
    def combinebalance(dict, balance_dict):
        for holder in dict:
            if holder in balance_dict:
                for account in dict[holder]['Accounts']:
                    if dict[holder]['Accounts'][account]['Product Type'] == 'SAV':
                        dict[holder]['Accounts'][account]['Balance'] = balance_dict[holder]['Balance']
        print('Combining balance information')
        return dict
    
    holder_dict = populateholder(holder_list)
    balance_dict = populatebalance(sav_list)
    dict = {}
    
    print('Populating Main Dict')
    for i in range(len(data_list)):
        date = pd.to_datetime(data_list[i][2]).date()    
        holder = data_list[i][0]
        account = data_list[i][1]
        transaction_id = data_list[i][16]
        channel = data_list[i][5]
        tr_details = transaction_details(data_list, i)
        acc_details = account_details(data_list, i)
        
        
        if week >= date.isocalendar()[1]:
            if holder in dict:
                if account not in dict[holder]['Accounts']:
                    dict[holder]['Accounts'][account]  = acc_details
                    dict[holder]['Accounts'][account]['Transactions'] = {}
                    if transaction_id not in dict[holder]['Accounts'][account]['Transactions']:
                        dict[holder]['Accounts'][account]['Transactions'][transaction_id]  = tr_details
                else:
                    dict[holder]['Accounts'][account]['Transactions'][transaction_id]  = tr_details

            else:
                dict[holder] = {'Channel': channel, 
                                'Accounts': {account: acc_details}}
                dict[holder]['Accounts'][account]['Transactions'] = {}
                dict[holder]['Accounts'][account]['Transactions'][transaction_id] = tr_details
    
    dict = combineholders(dict, holder_dict) 
    dict = combinebalance(dict, balance_dict)
    print('Dict Created')
    return dict

def create_rates(wb):
    def rank(array, r, cols, rows):
        curr = [1]*len(array[0])
        ref = array[r]

        for i in range(len(array)):
            if i != r and i in rows:
                for j in range(len(array[i])):
                    if j in cols and isinstance(array[i][j], float) and isinstance(ref[j], float):
                        if array[i][j] >= ref[j]:
                            curr[j] += 1
                    else:
                        pass
            else:
                pass
        for i in range(len(curr)):
            if i not in cols:
                curr[i] = -1
        return curr
    dict = {}
    for sheet in wb.sheets():
        name = sheet.name
        if name[-4:] == '2019':
            date = name[8:]
            year = date[-4:]
            if date[-7:-6] == ' ':
                day = date[-6:-5]
                month = date[:-7]
            else:
                day = date[-7:-5]
                month = date[:-8]       
            date = datetime.datetime.strptime(day+' '+month+' '+year, '%d %B %Y')
  
            dict[date] = {}
            data = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
            for i in range(len(data)):
                banks = [7,8,9,10,11,12,14,16,17,18,19,20,21,22,23,24,26,28,29,30,31,32,33,34,35,37,39,40,41]
                if date >= datetime.datetime(2019,4,25) and date < datetime.datetime(2019,5,15):
                    banks = [7,8,9,10,11,12,14,16,17,18,19,20,21,22,23,24,26,28,29,30,31,32,33,34,35,36,38,40,41,42]
                elif date>= datetime.datetime(2019,5,15) and date < datetime.datetime(2019,7,24):
                    banks = [7,8,9,10,11,12,14,16,17,18,19,20,21,22,23,24,26,28,29,30,31,32,33,34,35,36,37,39,41,42,43]
                elif date >= datetime.datetime(2019,7,24) and date < datetime.datetime(2019,11,19):
                    banks = [10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,27,29,31,32,33,34,35,36,37,38,39,40,42,44,45,46]
                elif date >= datetime.datetime(2019,11, 19):
                    banks = [10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,27,28,30,32,33,34,35,36,37,38,39,40,41,43,45,46,47]
        
                if i in banks:
                    if date < datetime.datetime(2019,9,26):
                        bank = data[i][1]
                    else:
                        bank = data[i][2]
                
                    dict[date][bank] = {}
                    if date < datetime.datetime(2019,9,26): 
                        accs = {2:'sav_rate',
                                4:'30_CSH',
                                5:'60_CSH', 
                                7:'30_GIC',
                                8:'60_GIC',
                                9:'90_GIC',
                                10:'120_GIC',
                                11:'180_GIC',
                                12:'270_GIC',
                                14:'1_GIC',
                                15:'HALF_GIC',
                                16:'2_GIC',
                                17:'3_GIC',
                                18:'4_GIC',
                                19:'5_GIC'}
                    else:
                        accs = {3:'sav_rate',
                                5:'30_CSH',
                                6:'60_CSH', 
                                8:'30_GIC',
                                9:'60_GIC',
                                10:'90_GIC',
                                11:'120_GIC',
                                12:'180_GIC',
                                13:'270_GIC',
                                15:'1_GIC',
                                16:'HALF_GIC',
                                17:'2_GIC',
                                18:'3_GIC',
                                19:'4_GIC',
                                20:'5_GIC'}
                    for n in accs.keys():
                        dict[date][bank][accs[n]] = data[i][n]

            home_trust = rank(data,banks[-3], accs.keys(), banks)
            home_bank = rank(data, banks[-2], accs.keys(), banks)
            oaken = rank(data, banks[-1], accs.keys(), banks)
        
            for i in range(0,3):
                if i == 0:
                    target = home_trust
                    label = 'HT_Rank'
                elif i == 1:
                    target = home_bank
                    label = 'HB_Rank'
                else:
                    target = oaken
                    label = 'Oaken_Rank'
                
                dict[date][label] = {}    
                for j in accs.keys():
                    dict[date][label][accs[j]] = target[j]
        else:
            pass
    return dict


def main(input_file_path = 'C:/Users/e-sshen/Desktop/GIC Data/FarhanExampleData.csv',
         holder_file_path = 'C:/Users/e-sshen/Desktop/GIC Data/AccountHolderExample.csv',
         sav_file_path = 'C:/Users/e-sshen/Desktop/GIC Data/FarhanPaymentSAV.csv',
         rates_file_path = 'C:/Users/e-sshen/Desktop/GIC Data/Competitor Rate Sheet 2019.xlsx'): 
    start_time = time.time()
    
    holders = pd.read_csv(holder_file_path)
    holder_list = holders.values.tolist()
      
    data = pd.read_csv(input_file_path)
    data[-1] = data.index    
    data_list = data.values.tolist()  
               
    sav = pd.read_csv(sav_file_path)
    sav_list = sav.values.tolist()  
    
    wb = xlrd.open_workbook(rates_file_path)    
    rates = create_rates(wb)
    
    arr = []
    dict = populatedict(data_list, holder_list, sav_list, 53)
    holder = aggregate(dict)
    for j in range(0,53):
        print(j)
        for i in holder:
            acc = []  
            if i.active_flag(j):
            #if i.holder == 80466:        
                acc.append(i.holder)
                acc.append(j)
                acc.append(float(i.age(j).days))
                acc.append(float(i.customer_age(j).days))
                acc.append(i.type)
                acc.append(i.channel)
                for m in i.total_product_vector():
                    acc.append(i.total_product_vector()[m])
                for n in i.active_product_vector(j):
                    acc.append(i.active_product_vector(j)[n])
                status, first = i.gic_sav_order()
                acc.append(status)
                acc.append(first)
                acc.append(i.average_contract(j))
                acc.append(i.average_interest(j))
                acc.append(i.average_term(j))
                for m in i.sav_char(j):
                    acc.append(m)
                for m in i.gic_char(j):
                    acc.append(m)
                if j == 0:
                    beg = datetime.datetime(2019,1,2)
                    for n in rates[beg]['Oaken']:
                        acc.append(rates[beg]['Oaken'][n])
                    for n in rates[beg]['Oaken_Rank']:
                        acc.append(rates[beg]['Oaken_Rank'][n])
                for x in rates:                        
                    if x.isocalendar()[1] == j:
                        for n in rates[x]['Oaken']:
                            acc.append(rates[x]['Oaken'][n])
                        for n in rates[x]['Oaken_Rank']:
                            acc.append(rates[x]['Oaken_Rank'][n])
                acc.append(i.gic_purchase(j))
                acc.append(i.gic_renewal(j))
                arr.append(acc)
                
    

    print('Converting Data to Dataframe')            
    df = pd.DataFrame(arr)
    df = df.replace(np.nan, '', regex=True)    
    print('Loading Data to .csv file')   
    print('Time to Run: ', round(time.time() - start_time, 2), ' Seconds')
    df.to_csv('C:/Users/e-sshen/Desktop/GIC Data/ActiveHolders.csv', float_format='%.2f', na_rep="NAN!")           
    
    
if __name__ == '__main__':
    main()    