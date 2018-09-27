'''
Created on 23 July 2018

@author: anthony
'''

import math


def point_color(red, green, blue):
    
    hexstr = "#{0:02x}{1:02x}{2:02x}".format(red, green, blue)
    
    return hexstr
    
                
def write_scatter_chart_html(name, title, xaxis_title, xaxis_log, yaxis_title, yaxis_log, data, htmlfile):
    
    DEFAULT_POINT_SIZE = 3
                
    with open(htmlfile, 'w') as html_file:
    
        html_file.write('<html>\n')
        html_file.write('  <head>\n')
        html_file.write('    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>\n')
        html_file.write('    <script type="text/javascript">\n')
        html_file.write('      google.charts.load("current", {"packages":["corechart"]});\n')
        html_file.write('      google.charts.setOnLoadCallback(drawChart);\n')
    
        html_file.write('      function drawChart() {\n')
        
        html_file.write('        var data = google.visualization.arrayToDataTable([\n')
        html_file.write('          ["' + xaxis_title + '", "' + yaxis_title + '", {"type": "string", "role": "style"}],\n')
        for dpoint in data:
            html_file.write('          [' + str(dpoint[0]) + ',' + str(dpoint[1]) 
                            + ', "point { size: ' + str(DEFAULT_POINT_SIZE * dpoint[2]) + '; fill-color: ' + point_color(dpoint[3], dpoint[4], dpoint[5])
                            + '}"],\n')
        html_file.write('        ]);\n')
        html_file.write('       var options = {\n')
#        html_file.write('          chart: {\n')
        html_file.write('            title: "' + title + '",\n')
        #html_file.write('            subtitle: "'+ origin_str_line + '",\n')
#        html_file.write('          },\n')
        html_file.write('          titleTextStyle: {fontSize: 20, bold: true},\n')
#        html_file.write('          reverseCategories: true,\n')
        html_file.write('            textStyle: {fontSize: 20},\n')
        html_file.write('          hAxis: {\n')
        html_file.write('            title: "' + xaxis_title + '",\n')
        html_file.write('            textStyle: {fontSize: 20},\n')
        if xaxis_log:
            html_file.write('            scaleType: "log",\n')
#         html_file.write('            minValue: -0.02,\n')
#         html_file.write('            maxValue: 1.00,\n')
#        html_file.write('            viewWindow: {min: -0.02, max: 1.01},\n')
#        html_file.write('            ticks: [0, 0.5, 1.0],\n')
        html_file.write('          },\n')
        html_file.write('          vAxis: {\n')
        html_file.write('            title: "' + yaxis_title + '",\n')
        html_file.write('            textStyle: {fontSize: 20},\n')
        if yaxis_log:
            html_file.write('            scaleType: "log",\n')
#         html_file.write('            minValue: -0.02,\n')
#         html_file.write('            maxValue: 1.00,\n')
#        html_file.write('            viewWindow: {min: -0.02, max: 1.01},\n')
#        html_file.write('            ticks: [0, 0.5, 1.0],\n')
        html_file.write('          },\n')
        html_file.write('          legend: { position: "none" },\n')
        html_file.write('        };\n')
        html_file.write('        var chart = new google.visualization.ScatterChart(document.getElementById("scatterchart_' + name + '"));\n')
        html_file.write('        chart.draw(data, options);\n')
        html_file.write('      }\n')
        html_file.write('    </script>\n')
        html_file.write('  </head>\n')
        html_file.write('  <body>\n')
        html_file.write('    <div id="scatterchart_' + name + '" style="width: 100%; height: 100%;"></div>\n')
        html_file.write('  </body>\n')
        html_file.write('</html>\n')


def write_bar_chart_html(name, units, bar_title, target_value, htmlfile, probabilities, prob_cutoff, class2value_method,
                         v_axis_direction, normalize=False):
    
    tot_prob = [0] * len(probabilities[0])
    for prob_set in probabilities:
        for classification in range(len(prob_set)):
            if prob_set[classification] >= prob_cutoff:
                tot_prob[classification] += prob_set[classification]
            
    prob_scale = 0.0
    nclass = len(tot_prob)
    for classification in range(nclass):
        if normalize:
            prob_scale = max(prob_scale, tot_prob[classification])
        else:
            prob_scale += tot_prob[classification]
    # TEST
    #prob_scale = len(prob_set)
            
    with open(htmlfile, 'w') as html_file:
    
        html_file.write('<html>\n')
        html_file.write('  <head>\n')
        html_file.write('    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>\n')
        html_file.write('    <script type="text/javascript">\n')
        html_file.write('      google.charts.load("current", {"packages":["bar"]});\n')
        html_file.write('      google.charts.setOnLoadCallback(drawChart);\n')
    
        html_file.write('      function drawChart() {\n')
        html_file.write('        var data = google.visualization.arrayToDataTable([\n')
        html_file.write('          ["' + name + ' ' + units + '", "Probability", "Target"],\n')
        value_low = class2value_method(0, nclass)
        for classification in range(nclass):
            value_high = class2value_method(classification + 1, nclass )
            value = (value_low + value_high) / 2.0
            prob_target_value = -0.02 if target_value is not None and target_value > value_low and target_value <= value_high else 0.0
            html_file.write('          [' + str(value) + ', '+ str(tot_prob[classification] / prob_scale) + ', '+ str(prob_target_value) + '],\n')
            value_low = value_high
        html_file.write('        ]);\n')
        html_file.write('       var options = {\n')
#        html_file.write('          chart: {\n')
        html_file.write('            title: "' + name + '",\n')
        #html_file.write('            subtitle: "'+ origin_str_line + '",\n')
#        html_file.write('          },\n')
        html_file.write('          titleTextStyle: {fontSize: 20, bold: true},\n')
#        html_file.write('          reverseCategories: true,\n')
        html_file.write('          bars: "horizontal",\n')
        html_file.write('          isStacked: true,\n')
        html_file.write('          series: { 0:{color: "blue", visibleInLegend: false}, 1:{color: "red", visibleInLegend: false}},\n')
        html_file.write('          bar: {\n')
        html_file.write('            groupWidth: "100%",\n')
        html_file.write('          },\n')
        html_file.write('          hAxis: {\n')
#         html_file.write('            minValue: -0.02,\n')
#         html_file.write('            maxValue: 1.00,\n')
        html_file.write('            viewWindow: {min: -0.02, max: 1.01},\n')
#        html_file.write('            ticks: [0, 0.5, 1.0],\n')
        html_file.write('            title: "' + bar_title + '",\n')
        html_file.write('            textStyle: {fontSize: 20},\n')
        html_file.write('          },\n')
        html_file.write('          vAxis: {\n')
        html_file.write('            direction: '+ str(v_axis_direction) + ',\n')
        view_low = class2value_method(0, nclass)
        view_high = class2value_method(nclass, nclass)
        if v_axis_direction > 0:
            html_file.write('            viewWindow: {min: '+ str(view_low) + ', max: '+ str(view_high) + '},\n')
        else:
            html_file.write('            viewWindow: {min: '+ str(view_high) + ', max: '+ str(view_low) + '},\n')
        html_file.write('            textStyle: {fontSize: 20},\n')
        html_file.write('          },\n')
        html_file.write('          legend: { position: "none" },\n')
        html_file.write('        };\n')
    
        html_file.write('        var chart = new google.charts.Bar(document.getElementById("barchart_' + name + '"));\n')
    
        html_file.write('        chart.draw(data, google.charts.Bar.convertOptions(options));\n')
        html_file.write('      }\n')
        html_file.write('    </script>\n')
        html_file.write('  </head>\n')
        html_file.write('  <body>\n')
        html_file.write('    <div id="barchart_' + name + '" style="width: 100%; height: 100%;"></div>\n')
        html_file.write('  </body>\n')
        html_file.write('</html>\n')


