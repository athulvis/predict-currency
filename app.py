import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan
from prophet import Prophet
import plotly
import plotly.graph_objs as go




def read_data(filename, parameter, period, splitval):    
    df = pd.read_csv(filename, index_col=None)
    df['Date'] = pd.to_datetime(df["Date"],infer_datetime_format=True)

    if parameter == 'Daily':

        df['Daily'] = df["Confirmed"].diff()
        new_df = df[["Date","Daily"]][splitval:]
        new_df = new_df.rename(columns={'Date':'ds', 'Daily':'y'})
        return new_df

    elif parameter == 'Hospitalized':

        new_df = df[["Date","Hosptilised"]][splitval:]
        new_df = new_df.rename(columns={'Date':'ds', 'Hosptilised':'y'})
        return new_df

    elif parameter == 'TPR':

        df["Daily"] = df["Confirmed"].diff()
        df['Daily_Tested'] = df["Tested"].diff()
        df["PR"] = df["Daily"]/df["Daily_Tested"]
        df["PR"] = df["PR"].replace(np.inf, 0)
        new_df = df[["Date","PR"]][splitval:]
        new_df = new_df.rename(columns={'Date':'ds', 'PR':'y'})
        return new_df

    else:
        print("wrong parameter. Please use one from Daily, Hosp, PR")


# In[5]:


def dist_data(filename, parameter, splitval, district):    
    df = pd.read_excel(filename, sheet_name = district)
    df['Date'] = pd.to_datetime(df["Date"],infer_datetime_format=True)

    if parameter == 'Daily':
        new_df = df[["Date","Confirmed"]][splitval:].dropna()
        new_df = new_df.rename(columns={'Date':'ds', 'Confirmed':'y'})
        return new_df

def district_data(filename, parameter, splitval, district):    
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df["Date"],infer_datetime_format=True)

    if parameter == 'Daily':
        if district == 'Kerala':
            new_df = df[["Date","Kerala"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'Kerala':'y'})
            return new_df
        elif district == 'TVM':
            new_df = df[["Date","TVM"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'TVM':'y'})
            return new_df
        elif district == 'KLM':
            new_df = df[["Date","KLM"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'KLM':'y'})
            return new_df
            
        elif district == 'PTA':
            new_df = df[["Date","PTA"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'PTA':'y'})
            return new_df
        
        elif district == 'ALP':
            new_df = df[["Date","ALP"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'ALP':'y'})
            return new_df
            
        elif district == 'KTM':
            new_df = df[["Date","KTM"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'KTM':'y'})
            return new_df
            
        elif district == 'IDK':
            new_df = df[["Date","IDK"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'IDK':'y'})
            return new_df
            
        elif district == 'EKM':
            new_df = df[["Date","EKM"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'EKM':'y'})
            return new_df
            
        elif district == 'TSR':
            new_df = df[["Date","TSR"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'TSR':'y'})
            return new_df
            
        elif district == 'PKD':
            new_df = df[["Date","PKD"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'PKD':'y'})
            return new_df
        
        elif district == 'MLP':
            new_df = df[["Date","MLP"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'MLP':'y'})
            return new_df
            
        elif district == 'KKD':
            new_df = df[["Date","KKD"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'KKD':'y'})
            return new_df
            
        elif district == 'WYD':
            new_df = df[["Date","WYD"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'WYD':'y'})
            return new_df
            
        elif district == 'KNR':
            new_df = df[["Date","KNR"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'KNR':'y'})
            return new_df
        elif district == 'KSD':
            new_df = df[["Date","KSD"]][splitval:].dropna()
            new_df = new_df.rename(columns={'Date':'ds', 'KSD':'y'})
            return new_df
        else:
            print("Enter invaid value")


#def model_prediction(self, df, frequency, seasonality):
#m = Prophet() #normal case uses additive model with auto identifed seasonality

def predictfn(df,period, frequency = 'D'):
    m = Prophet(seasonality_mode= 'multiplicative', weekly_seasonality=True)
    model = m.fit(df)
    seasonalities = model.seasonalities
    future_df = model.make_future_dataframe(periods=period, freq = frequency)
    forecast = model.predict(future_df)
    return forecast



# Create the plotly figure
def final_plot(df, forecast, district):
    parameter = 'Daily'
    period = 14
    splitval1 = 450
    splitval2 = 300

    dist_list = ['Kerala', 'TVM', 'KLM', 'PTA','ALP','KTM','IDK','EKM','TSR','PKD','MLP','KKD', 'WYD','KNR','KSD']

    title_d = district

    yhat = go.Scatter(x = forecast['ds'], y = forecast['yhat'], mode = 'lines', marker = {'color': '#3bbed7'},
                      line = {'width': 3}, name = 'Forecast',)

    yhat_lower = go.Scatter(x = forecast['ds'], y = forecast['yhat_lower'], marker = {'color': 'rgba(30,129,176,0.75)'},
                              showlegend = False, hoverinfo = 'none',)

    yhat_upper = go.Scatter(x = forecast['ds'], y = forecast['yhat_upper'], fill='tonexty',
                            fillcolor = 'rgba(30,129,176,0.75)', name = 'Confidence', hoverinfo='none',mode = 'none')

    actual = go.Scatter(x = df['ds'], y = df['y'], mode = 'markers', marker = {'color': '#21130d','size': 4,
        'line': {'color': '#000000','width': .75}}, name = 'Actual')

    layout = go.Layout(yaxis = {'title': parameter, 'tickformat': format('y'), 'hoverformat': format('y')},
                hovermode = 'x', xaxis = { 'title': 'Date'}, margin = {'t': 20,'b': 50,'l': 60,'r': 10},
                  legend = {'bgcolor': 'rgba(0,0,0,0)'}, title=f"Covid19 {title_d}")

    data = [yhat_lower, yhat_upper, yhat, actual]
    
    return data, layout



import dash
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash()
server = app.server

dist_list = ['Kerala', 'TVM', 'KLM', 'PTA','ALP','KTM','IDK','EKM','TSR','PKD','MLP','KKD','WYD','KNR','KSD']
fig_name = dist_list
fig_dropdown = html.Div([
    dcc.Dropdown(
        id='fig_dropdown',
        options=[{'label': x, 'value': x} for x in fig_name],
        value=None
    )])
fig_plot = html.Div(id='fig_plot')
app.layout = html.Div([fig_dropdown, fig_plot])

@app.callback(
dash.dependencies.Output('fig_plot', 'children'),
[dash.dependencies.Input('fig_dropdown', 'value')])
def update_output(fig_name):
    return name_to_figure(fig_name)


def name_to_figure(fig_name):
    
    parameter = 'Daily'
    period = 14
    splitval1 = 300
    splitval2 = 300
    filename = 'data/district_wise.csv'
    figure = go.Figure()
    
        
    if fig_name == dist_list[1]:
        dist1_df = district_data(filename, parameter, splitval2, dist_list[1])
        forecast1 = predictfn(dist1_df,period, frequency = 'D')
        data1, layout1 = final_plot(dist1_df, forecast1, dist_list[1])
        fig = dict(data = data1, layout = layout1)
        
    elif fig_name == dist_list[2]: 
        dist2_df = district_data(filename, parameter, splitval2, dist_list[2])
        forecast2 = predictfn(dist2_df,period, frequency = 'D')
        data2, layout2 = final_plot(dist2_df, forecast2, dist_list[2])
        fig = dict(data = data2, layout = layout2)
    
    elif fig_name == dist_list[3]: 
        dist3_df = district_data(filename, parameter, splitval2, dist_list[3])
        forecast3 = predictfn(dist3_df,period, frequency = 'D')
        data3, layout3 = final_plot(dist3_df, forecast3, dist_list[3])
        fig = dict(data = data3, layout = layout3)
        
    elif fig_name == dist_list[4]: 
        dist4_df = district_data(filename, parameter, splitval2, dist_list[4])
        forecast4 = predictfn(dist4_df,period, frequency = 'D')
        data4, layout4 = final_plot(dist4_df, forecast4, dist_list[4])
        fig = dict(data = data4, layout = layout4)
    
    elif fig_name == dist_list[5]: 
        dist5_df = district_data(filename, parameter, splitval2, dist_list[5])
        forecast5 = predictfn(dist5_df,period, frequency = 'D')
        data5, layout5 = final_plot(dist5_df, forecast5, dist_list[5])
        fig = dict(data = data5, layout = layout5)
        
    elif fig_name == dist_list[6]: 
        dist6_df = district_data(filename, parameter, splitval2, dist_list[6])
        forecast6 = predictfn(dist6_df,period, frequency = 'D')
        data6, layout6 = final_plot(dist6_df, forecast6, dist_list[6])
        fig = dict(data = data6, layout = layout6)
        
    elif fig_name == dist_list[7]: 
        dist7_df = district_data(filename, parameter, splitval2, dist_list[7])
        forecast7 = predictfn(dist7_df,period, frequency = 'D')
        data7, layout7 = final_plot(dist7_df, forecast7, dist_list[7])
        fig = dict(data = data7, layout = layout7)
        
    elif fig_name == dist_list[8]: 
        dist8_df = district_data(filename, parameter, splitval2, dist_list[8])
        forecast8 = predictfn(dist8_df,period, frequency = 'D')
        data8, layout8 = final_plot(dist8_df, forecast8, dist_list[8])
        fig = dict(data = data8, layout = layout8)
        
    elif fig_name == dist_list[9]: 
        dist9_df = district_data(filename, parameter, splitval2, dist_list[9])
        forecast9 = predictfn(dist9_df,period, frequency = 'D')
        data9, layout9 = final_plot(dist9_df, forecast9, dist_list[9])
        fig = dict(data = data9, layout = layout9)
        
    elif fig_name == dist_list[10]: 
        dist10_df = district_data(filename, parameter, splitval2, dist_list[10])
        forecast10 = predictfn(dist10_df,period, frequency = 'D')
        data10, layout10 = final_plot(dist10_df, forecast10, dist_list[10])
        fig = dict(data = data10, layout = layout10)
    
    elif fig_name == dist_list[11]: 
        dist11_df = district_data(filename, parameter, splitval2, dist_list[11])
        forecast11 = predictfn(dist11_df,period, frequency = 'D')
        data11, layout11 = final_plot(dist11_df, forecast11, dist_list[11])
        fig = dict(data = data11, layout = layout11)
        
    elif fig_name == dist_list[12]: 
        dist12_df = district_data(filename, parameter, splitval2, dist_list[12])
        forecast12 = predictfn(dist12_df,period, frequency = 'D')
        data12, layout12 = final_plot(dist12_df, forecast12, dist_list[12])
        fig = dict(data = data12, layout = layout12)
        
    elif fig_name == dist_list[13]: 
        dist13_df = district_data(filename, parameter, splitval2, dist_list[13])
        forecast13 = predictfn(dist13_df,period, frequency = 'D')
        data13, layout13 = final_plot(dist13_df, forecast13, dist_list[13])
        fig = dict(data = data13, layout = layout13)
    
    elif fig_name == dist_list[14]: 
        dist14_df = district_data(filename, parameter, splitval2, dist_list[14])
        forecast14 = predictfn(dist14_df,period, frequency = 'D')
        data14, layout14 = final_plot(dist14_df,forecast14, dist_list[14])
        fig = dict(data = data14, layout = layout14)

    
    else:
        dist0_df = district_data(filename, parameter, splitval2, dist_list[0])
        forecast0 = predictfn(dist0_df,period, frequency = 'D')
        data0, layout0 = final_plot(dist0_df, forecast0, dist_list[0])
        fig = dict(data = data0, layout = layout0)

        
    return dcc.Graph(figure=fig)

if __name__== "__main__":
    app.run_server(debug=True) #, use_reloader=False)





