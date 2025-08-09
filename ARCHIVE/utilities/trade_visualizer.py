#!/usr/bin/env python3
"""
Enhanced Trading Visualizer - Web server and real-time monitoring
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import time
from flask import Flask, render_template_string, jsonify, send_file, request
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import plotly.io as pio
import webbrowser
from rich.console import Console

console = Console()

class TradingVisualizer:
    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self.setup_routes()
        self.live_data = {
            'trades': [],
            'positions': [],
            'equity': [],
            'prices': [],
            'timestamps': []
        }
        self.backtest_results = None
        self.is_monitoring = False
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/live_data')
        def get_live_data():
            return jsonify(self.live_data)
        
        @self.app.route('/api/backtest_data')
        def get_backtest_data():
            if self.backtest_results is not None:
                return jsonify({
                    'timestamps': self.backtest_results['timestamp'].tolist(),
                    'prices': self.backtest_results['price'].tolist(),
                    'positions': self.backtest_results.get('position', []).tolist(),
                    'equity': self.backtest_results.get('equity', []).tolist(),
                    'trades': self.get_trades_from_backtest()
                })
            return jsonify({})
        
        @self.app.route('/api/export/<format>')
        def export_visualization(format):
            """Export current visualization"""
            if format == 'html':
                return self.export_as_html()
            elif format == 'png':
                return self.export_as_png()
            elif format == 'pdf':
                return self.export_as_pdf()
            elif format == 'csv':
                return self.export_as_csv()
            return jsonify({'error': 'Invalid format'}), 400
    
    def get_dashboard_template(self):
        """Get HTML template for dashboard"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container {
            margin: 20px 0;
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            min-width: 150px;
        }
        .stat-label {
            font-size: 14px;
            color: #888;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }
        .positive { color: #00ff00; }
        .negative { color: #ff0000; }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }
        button:hover {
            background: #0056b3;
        }
        .export-btn {
            background: #28a745;
        }
        .export-btn:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Trading Bot Dashboard</h1>
        <div id="status">Loading...</div>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-label">Current Price</div>
            <div class="stat-value" id="current-price">-</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Position</div>
            <div class="stat-value" id="current-position">0</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Equity</div>
            <div class="stat-value" id="current-equity">$10,000</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">P&L</div>
            <div class="stat-value" id="current-pnl">0%</div>
        </div>
    </div>
    
    <div class="controls">
        <button onclick="toggleAutoUpdate()">Toggle Auto-Update</button>
        <button onclick="refreshData()">Refresh</button>
        <button class="export-btn" onclick="exportChart('html')">Export HTML</button>
        <button class="export-btn" onclick="exportChart('png')">Export PNG</button>
        <button class="export-btn" onclick="exportChart('pdf')">Export PDF</button>
        <button class="export-btn" onclick="exportChart('csv')">Export CSV</button>
    </div>
    
    <div class="chart-container">
        <div id="price-chart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <div id="position-chart" style="height: 200px;"></div>
    </div>
    
    <div class="chart-container">
        <div id="equity-chart" style="height: 300px;"></div>
    </div>
    
    <script>
        let autoUpdate = true;
        let updateInterval = null;
        
        function initCharts() {
            // Price chart layout
            Plotly.newPlot('price-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Price',
                line: {color: '#00bfff'}
            }], {
                title: 'Price & Trades',
                paper_bgcolor: '#2a2a2a',
                plot_bgcolor: '#1a1a1a',
                font: {color: '#ffffff'},
                xaxis: {gridcolor: '#444'},
                yaxis: {gridcolor: '#444', title: 'Price ($)'}
            });
            
            // Position chart
            Plotly.newPlot('position-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                fill: 'tozeroy',
                name: 'Position',
                line: {color: '#ffff00'}
            }], {
                title: 'Position',
                paper_bgcolor: '#2a2a2a',
                plot_bgcolor: '#1a1a1a',
                font: {color: '#ffffff'},
                xaxis: {gridcolor: '#444'},
                yaxis: {gridcolor: '#444', title: 'Position Size'}
            });
            
            // Equity chart
            Plotly.newPlot('equity-chart', [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Equity',
                line: {color: '#00ff00'}
            }], {
                title: 'Equity Curve',
                paper_bgcolor: '#2a2a2a',
                plot_bgcolor: '#1a1a1a',
                font: {color: '#ffffff'},
                xaxis: {gridcolor: '#444'},
                yaxis: {gridcolor: '#444', title: 'Equity ($)'}
            });
        }
        
        function updateCharts(data) {
            if (!data || !data.timestamps) return;
            
            // Update price chart
            Plotly.update('price-chart', {
                x: [data.timestamps],
                y: [data.prices]
            }, {}, [0]);
            
            // Add trade markers
            if (data.trades && data.trades.length > 0) {
                const buyTrades = data.trades.filter(t => t.action === 'buy');
                const sellTrades = data.trades.filter(t => t.action === 'sell');
                
                if (buyTrades.length > 0) {
                    Plotly.addTraces('price-chart', {
                        x: buyTrades.map(t => t.timestamp),
                        y: buyTrades.map(t => t.price),
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Buy',
                        marker: {
                            size: 12,
                            color: '#00ff00',
                            symbol: 'triangle-up'
                        }
                    });
                }
                
                if (sellTrades.length > 0) {
                    Plotly.addTraces('price-chart', {
                        x: sellTrades.map(t => t.timestamp),
                        y: sellTrades.map(t => t.price),
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Sell',
                        marker: {
                            size: 12,
                            color: '#ff0000',
                            symbol: 'triangle-down'
                        }
                    });
                }
            }
            
            // Update position chart
            Plotly.update('position-chart', {
                x: [data.timestamps],
                y: [data.positions]
            }, {}, [0]);
            
            // Update equity chart
            Plotly.update('equity-chart', {
                x: [data.timestamps],
                y: [data.equity]
            }, {}, [0]);
            
            // Update stats
            if (data.prices.length > 0) {
                const lastPrice = data.prices[data.prices.length - 1];
                const lastPosition = data.positions[data.positions.length - 1] || 0;
                const lastEquity = data.equity[data.equity.length - 1] || 10000;
                const pnl = ((lastEquity - 10000) / 10000 * 100).toFixed(2);
                
                document.getElementById('current-price').textContent = `$${lastPrice.toFixed(2)}`;
                document.getElementById('current-position').textContent = lastPosition.toFixed(4);
                document.getElementById('current-equity').textContent = `$${lastEquity.toFixed(2)}`;
                
                const pnlElement = document.getElementById('current-pnl');
                pnlElement.textContent = `${pnl}%`;
                pnlElement.className = parseFloat(pnl) >= 0 ? 'stat-value positive' : 'stat-value negative';
            }
            
            document.getElementById('status').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        }
        
        async function refreshData() {
            try {
                // Try live data first
                const liveResponse = await fetch('/api/live_data');
                const liveData = await liveResponse.json();
                
                if (liveData.timestamps && liveData.timestamps.length > 0) {
                    updateCharts(liveData);
                } else {
                    // Fall back to backtest data
                    const backtestResponse = await fetch('/api/backtest_data');
                    const backtestData = await backtestResponse.json();
                    updateCharts(backtestData);
                }
            } catch (error) {
                console.error('Error fetching data:', error);
                document.getElementById('status').textContent = 'Error loading data';
            }
        }
        
        function toggleAutoUpdate() {
            autoUpdate = !autoUpdate;
            if (autoUpdate) {
                updateInterval = setInterval(refreshData, 1000);
            } else {
                clearInterval(updateInterval);
            }
        }
        
        async function exportChart(format) {
            try {
                const response = await fetch(`/api/export/${format}`);
                if (format === 'csv') {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `trading_data_${new Date().toISOString()}.csv`;
                    a.click();
                } else {
                    const result = await response.json();
                    if (result.filename) {
                        window.open(result.filename, '_blank');
                    }
                }
            } catch (error) {
                console.error('Export error:', error);
            }
        }
        
        // Initialize
        initCharts();
        refreshData();
        if (autoUpdate) {
            updateInterval = setInterval(refreshData, 1000);
        }
    </script>
</body>
</html>
        '''
    
    def get_trades_from_backtest(self):
        """Extract trades from backtest results"""
        if self.backtest_results is None:
            return []
        
        trades = []
        df = self.backtest_results
        
        # Find rows where action is buy or sell
        for idx, row in df.iterrows():
            if 'action' in df.columns and row['action'] in ['buy', 'sell']:
                trades.append({
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'action': row['action'],
                    'size': row.get('size', 0)
                })
        
        return trades
    
    def export_as_html(self):
        """Export current visualization as HTML"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'export_dashboard_{timestamp}.html'
        
        # Create standalone HTML with current data
        if self.backtest_results is not None:
            fig = self.create_plotly_figure(self.backtest_results)
            fig.write_html(filename)
            return jsonify({'filename': filename, 'status': 'success'})
        
        return jsonify({'error': 'No data to export'}), 400
    
    def export_as_png(self):
        """Export current visualization as PNG"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'export_chart_{timestamp}.png'
        
        if self.backtest_results is not None:
            fig = self.create_plotly_figure(self.backtest_results)
            fig.write_image(filename, width=1200, height=900)
            return jsonify({'filename': filename, 'status': 'success'})
        
        return jsonify({'error': 'No data to export'}), 400
    
    def export_as_pdf(self):
        """Export current visualization as PDF"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'export_report_{timestamp}.pdf'
        
        if self.backtest_results is not None:
            fig = self.create_plotly_figure(self.backtest_results)
            fig.write_image(filename, format='pdf', width=1200, height=900)
            return jsonify({'filename': filename, 'status': 'success'})
        
        return jsonify({'error': 'No data to export'}), 400
    
    def export_as_csv(self):
        """Export data as CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'export_data_{timestamp}.csv'
        
        if self.backtest_results is not None:
            self.backtest_results.to_csv(filename, index=False)
            return send_file(filename, as_attachment=True)
        
        return jsonify({'error': 'No data to export'}), 400
    
    def create_plotly_figure(self, df):
        """Create comprehensive Plotly figure"""
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Trades', 'Position', 'Equity Curve'),
            row_heights=[0.5, 0.2, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price'], 
                      name='Price', line=dict(color='lightblue')),
            row=1, col=1
        )
        
        # Add trade markers if available
        if 'action' in df.columns:
            buys = df[df['action'] == 'buy']
            sells = df[df['action'] == 'sell']
            
            if len(buys) > 0:
                fig.add_trace(
                    go.Scatter(x=buys['timestamp'], y=buys['price'],
                              mode='markers', name='Buy',
                              marker=dict(color='green', size=12, symbol='triangle-up')),
                    row=1, col=1
                )
            
            if len(sells) > 0:
                fig.add_trace(
                    go.Scatter(x=sells['timestamp'], y=sells['price'],
                              mode='markers', name='Sell',
                              marker=dict(color='red', size=12, symbol='triangle-down')),
                    row=1, col=1
                )
        
        # Position chart
        if 'position' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['position'],
                          name='Position', fill='tozeroy', line=dict(color='yellow')),
                row=2, col=1
            )
        
        # Equity chart
        if 'equity' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['equity'],
                          name='Equity', line=dict(color='green')),
                row=3, col=1
            )
        
        fig.update_layout(
            height=900,
            title_text="Trading Bot Performance",
            template='plotly_dark'
        )
        
        return fig
    
    def load_backtest_results(self, filename):
        """Load backtest results from file"""
        if os.path.exists(filename):
            self.backtest_results = pd.read_csv(filename)
            console.print(f"[green]Loaded backtest results: {filename}[/green]")
        else:
            console.print(f"[red]File not found: {filename}[/red]")
    
    def monitor_live_file(self, filename, interval=1):
        """Monitor a live results file for updates"""
        self.is_monitoring = True
        console.print(f"[yellow]Monitoring {filename} for updates...[/yellow]")
        
        def monitor_loop():
            last_modified = 0
            while self.is_monitoring:
                try:
                    if os.path.exists(filename):
                        current_modified = os.path.getmtime(filename)
                        if current_modified > last_modified:
                            # File has been updated
                            df = pd.read_csv(filename)
                            
                            # Update live data
                            if len(df) > 0:
                                self.live_data = {
                                    'timestamps': df['timestamp'].tolist()[-100:],  # Last 100 points
                                    'prices': df['price'].tolist()[-100:],
                                    'positions': df.get('position', []).tolist()[-100:],
                                    'equity': df.get('equity', []).tolist()[-100:],
                                    'trades': self.get_trades_from_dataframe(df)
                                }
                            
                            last_modified = current_modified
                    
                    time.sleep(interval)
                except Exception as e:
                    console.print(f"[red]Monitor error: {e}[/red]")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def get_trades_from_dataframe(self, df):
        """Extract recent trades from dataframe"""
        trades = []
        if 'action' in df.columns:
            recent_trades = df[df['action'].isin(['buy', 'sell'])].tail(20)
            for _, row in recent_trades.iterrows():
                trades.append({
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'action': row['action'],
                    'size': row.get('size', 0)
                })
        return trades
    
    def start_server(self, open_browser=True):
        """Start the Flask web server"""
        console.print(f"\n[green]Starting visualization server on http://localhost:{self.port}[/green]")
        console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")
        
        if open_browser:
            # Open browser after a short delay
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f'http://localhost:{self.port}')
            
            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.start()
        
        # Run Flask app
        self.app.run(host='0.0.0.0', port=self.port, debug=False)
    
    def stop_monitoring(self):
        """Stop monitoring live files"""
        self.is_monitoring = False


# CLI Integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Visualizer Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--backtest', help='Backtest results file to load')
    parser.add_argument('--live', help='Live results file to monitor')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    visualizer = TradingVisualizer(port=args.port)
    
    if args.backtest:
        visualizer.load_backtest_results(args.backtest)
    
    if args.live:
        visualizer.monitor_live_file(args.live)
    
    try:
        visualizer.start_server(open_browser=not args.no_browser)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
        visualizer.stop_monitoring()