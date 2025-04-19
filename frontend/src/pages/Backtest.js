import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Divider,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import { Line, Bar } from 'react-chartjs-2';
import { runBacktest } from '../store/backtestSlice';
import { fetchModels } from '../store/modelsSlice';

const Backtest = () => {
  const dispatch = useDispatch();
  const { results, loading, error } = useSelector((state) => state.backtest);
  const { ensembles, agents } = useSelector((state) => state.models);
  const [tabValue, setTabValue] = useState(0);
  const [formData, setFormData] = useState({
    strategyType: 'combined',
    symbol: 'BTC/USDT',
    timeframe: '1h',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialBalance: 10000,
    ensembleId: '',
    agentId: '',
    ensembleWeight: 0.5,
    rlWeight: 0.5,
    positionSize: 0.1,
    stopLoss: 0.05,
    takeProfit: 0.1,
    transactionFee: 0.001
  });

  useEffect(() => {
    dispatch(fetchModels());
  }, [dispatch]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    if (name === 'ensembleWeight') {
      const newValue = parseFloat(value);
      setFormData({
        ...formData,
        ensembleWeight: newValue,
        rlWeight: 1 - newValue
      });
    } else if (name === 'rlWeight') {
      const newValue = parseFloat(value);
      setFormData({
        ...formData,
        rlWeight: newValue,
        ensembleWeight: 1 - newValue
      });
    } else {
      setFormData({
        ...formData,
        [name]: value
      });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Transform form data to API format
    const backtestConfig = {
      strategy: {
        type: formData.strategyType,
        symbol: formData.symbol,
        timeframe: formData.timeframe,
        ensemble_id: formData.ensembleId,
        agent_id: formData.agentId,
        ensemble_weight: parseFloat(formData.ensembleWeight),
        rl_weight: parseFloat(formData.rlWeight),
        position_size: parseFloat(formData.positionSize),
        stop_loss_pct: parseFloat(formData.stopLoss),
        take_profit_pct: parseFloat(formData.takeProfit)
      },
      backtest_params: {
        start_date: formData.startDate,
        end_date: formData.endDate,
        initial_balance: parseFloat(formData.initialBalance),
        transaction_fee: parseFloat(formData.transactionFee)
      }
    };
    
    dispatch(runBacktest(backtestConfig));
  };

  // Prepare chart data for equity curve
  const prepareEquityCurveData = () => {
    if (!results || !results.equity_curve) return null;

    return {
      labels: results.equity_curve.map(point => new Date(point.timestamp).toLocaleDateString()),
      datasets: [
        {
          label: 'Equity',
          data: results.equity_curve.map(point => point.equity),
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
          fill: false,
        },
      ],
    };
  };

  // Prepare chart data for trades
  const prepareTradesData = () => {
    if (!results || !results.trades) return null;

    return {
      labels: results.trades.map(trade => new Date(trade.timestamp).toLocaleDateString()),
      datasets: [
        {
          label: 'Profit/Loss',
          data: results.trades.map(trade => trade.pnl),
          backgroundColor: results.trades.map(trade => 
            trade.pnl >= 0 ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
          ),
          borderColor: results.trades.map(trade => 
            trade.pnl >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
          ),
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare chart data for monthly returns
  const prepareMonthlyReturnsData = () => {
    if (!results || !results.monthly_returns) return null;

    return {
      labels: Object.keys(results.monthly_returns),
      datasets: [
        {
          label: 'Monthly Returns (%)',
          data: Object.values(results.monthly_returns),
          backgroundColor: Object.values(results.monthly_returns).map(value => 
            value >= 0 ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
          ),
          borderColor: Object.values(results.monthly_returns).map(value => 
            value >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
          ),
          borderWidth: 1,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Backtest Results',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  const barChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Trade Results',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const equityCurveData = prepareEquityCurveData();
  const tradesData = prepareTradesData();
  const monthlyReturnsData = prepareMonthlyReturnsData();

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom component="div">
        Backtest
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Backtest Configuration
            </Typography>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel id="strategyType-label">Strategy Type</InputLabel>
                    <Select
                      labelId="strategyType-label"
                      id="strategyType"
                      name="strategyType"
                      value={formData.strategyType}
                      onChange={handleChange}
                      label="Strategy Type"
                    >
                      <MenuItem value="ensemble">Ensemble Learning</MenuItem>
                      <MenuItem value="rl">Reinforcement Learning</MenuItem>
                      <MenuItem value="combined">Combined (Ensemble + RL)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Trading Pair"
                    name="symbol"
                    value={formData.symbol}
                    onChange={handleChange}
                    margin="normal"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel id="timeframe-label">Timeframe</InputLabel>
                    <Select
                      labelId="timeframe-label"
                      id="timeframe"
                      name="timeframe"
                      value={formData.timeframe}
                      onChange={handleChange}
                      label="Timeframe"
                    >
                      <MenuItem value="1m">1 minute</MenuItem>
                      <MenuItem value="5m">5 minutes</MenuItem>
                      <MenuItem value="15m">15 minutes</MenuItem>
                      <MenuItem value="30m">30 minutes</MenuItem>
                      <MenuItem value="1h">1 hour</MenuItem>
                      <MenuItem value="4h">4 hours</MenuItem>
                      <MenuItem value="1d">1 day</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Start Date"
                    name="startDate"
                    type="date"
                    value={formData.startDate}
                    onChange={handleChange}
                    margin="normal"
                    InputLabelProps={{
                      shrink: true,
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="End Date"
                    name="endDate"
                    type="date"
                    value={formData.endDate}
                    onChange={handleChange}
                    margin="normal"
                    InputLabelProps={{
                      shrink: true,
                    }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Initial Balance"
                    name="initialBalance"
                    type="number"
                    value={formData.initialBalance}
                    onChange={handleChange}
                    margin="normal"
                    InputProps={{
                      startAdornment: '$',
                    }}
                  />
                </Grid>

                {(formData.strategyType === 'ensemble' || formData.strategyType === 'combined') && (
                  <Grid item xs={12}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel id="ensembleId-label">Ensemble Model</InputLabel>
                      <Select
                        labelId="ensembleId-label"
                        id="ensembleId"
                        name="ensembleId"
                        value={formData.ensembleId}
                        onChange={handleChange}
                        label="Ensemble Model"
                      >
                        {ensembles.map((model) => (
                          <MenuItem key={model.id} value={model.id}>
                            {model.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                )}

                {(formData.strategyType === 'rl' || formData.strategyType === 'combined') && (
                  <Grid item xs={12}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel id="agentId-label">RL Agent</InputLabel>
                      <Select
                        labelId="agentId-label"
                        id="agentId"
                        name="agentId"
                        value={formData.agentId}
                        onChange={handleChange}
                        label="RL Agent"
                      >
                        {agents.map((agent) => (
                          <MenuItem key={agent.id} value={agent.id}>
                            {agent.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                )}

                {formData.strategyType === 'combined' && (
                  <>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Ensemble Weight"
                        name="ensembleWeight"
                        type="number"
                        value={formData.ensembleWeight}
                        onChange={handleChange}
                        margin="normal"
                        inputProps={{ step: 0.1, min: 0, max: 1 }}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="RL Weight"
                        name="rlWeight"
                        type="number"
                        value={formData.rlWeight}
                        onChange={handleChange}
                        margin="normal"
                        inputProps={{ step: 0.1, min: 0, max: 1 }}
                      />
                    </Grid>
                  </>
                )}

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Position Size"
                    name="positionSize"
                    type="number"
                    value={formData.positionSize}
                    onChange={handleChange}
                    margin="normal"
                    helperText="Fraction of capital per trade"
                    inputProps={{ step: 0.05, min: 0.01, max: 1 }}
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Stop Loss"
                    name="stopLoss"
                    type="number"
                    value={formData.stopLoss}
                    onChange={handleChange}
                    margin="normal"
                    inputProps={{ step: 0.01, min: 0.01, max: 0.5 }}
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Take Profit"
                    name="takeProfit"
                    type="number"
                    value={formData.takeProfit}
                    onChange={handleChange}
                    margin="normal"
                    inputProps={{ step: 0.01, min: 0.01, max: 0.5 }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Transaction Fee"
                    name="transactionFee"
                    type="number"
                    value={formData.transactionFee}
                    onChange={handleChange}
                    margin="normal"
                    inputProps={{ step: 0.0001, min: 0, max: 0.01 }}
                  />
                </Grid>

                <Grid item xs={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    fullWidth
                    disabled={loading}
                    sx={{ mt: 2 }}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Run Backtest'}
                  </Button>
                </Grid>
              </Grid>
            </form>
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error.message}
            </Alert>
          )}

          {results ? (
            <Paper sx={{ p: 3 }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="backtest results tabs">
                  <Tab label="Summary" />
                  <Tab label="Equity Curve" />
                  <Tab label="Trades" />
                  <Tab label="Monthly Returns" />
                </Tabs>
              </Box>

              {/* Summary Tab */}
              {tabValue === 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Backtest Results Summary
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6}>
                      <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
                        <Typography variant="subtitle1" gutterBottom>
                          Performance Metrics
                        </Typography>
                        <Divider sx={{ mb: 2 }} />
                        <Grid container spacing={1}>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Initial Balance:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              ${results.initial_balance.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Final Balance:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              ${results.final_balance.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Total Return:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography
                              variant="body2"
                              color={results.total_return >= 0 ? 'success.main' : 'error.main'}
                            >
                              ${results.total_return.toFixed(2)} ({results.total_return_pct.toFixed(2)}%)
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Annualized Return:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography
                              variant="body2"
                              color={results.annualized_return >= 0 ? 'success.main' : 'error.main'}
                            >
                              {results.annualized_return.toFixed(2)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Max Drawdown:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="error.main">
                              {(results.max_drawdown * 100).toFixed(2)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Sharpe Ratio:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              {results.sharpe_ratio.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Sortino Ratio:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              {results.sortino_ratio.toFixed(2)}
                            </Typography>
                          </Grid>
                        </Grid>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
                        <Typography variant="subtitle1" gutterBottom>
                          Trade Statistics
                        </Typography>
                        <Divider sx={{ mb: 2 }} />
                        <Grid container spacing={1}>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Total Trades:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              {results.total_trades}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Win Rate:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              {(results.win_rate * 100).toFixed(2)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Profit Factor:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2">
                              {results.profit_factor.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Average Win:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="success.main">
                              ${results.avg_win.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Average Loss:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="error.main">
                              ${results.avg_loss.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Largest Win:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="success.main">
                              ${results.largest_win.toFixed(2)}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Largest Loss:
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="error.main">
                              ${results.largest_loss.toFixed(2)}
                            </Typography>
                          </Grid>
                        </Grid>
                      </Paper>
                    </Grid>
                  </Grid>
                </Box>
              )}

              {/* Equity Curve Tab */}
              {tabValue === 1 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Equity Curve
                  </Typography>
                  {equityCurveData ? (
                    <Box sx={{ height: 400 }}>
                      <Line options={chartOptions} data={equityCurveData} />
                    </Box>
                  ) : (
                    <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 5 }}>
                      No equity curve data available.
                    </Typography>
                  )}
                </Box>
              )}

              {/* Trades Tab */}
              {tabValue === 2 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Trade Results
                  </Typography>
                  {tradesData ? (
                    <>
                      <Box sx={{ height: 300, mb: 4 }}>
                        <Bar options={barChartOptions} data={tradesData} />
                      </Box>
                      <Typography variant="subtitle1" gutterBottom>
                        Recent Trades
                      </Typography>
                      <Box sx={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Date</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Type</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Entry Price</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Exit Price</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>Size</th>
                              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>P&L</th>
                            </tr>
                          </thead>
                          <tbody>
                            {results.trades.slice(0, 10).map((trade, index) => (
                              <tr key={index}>
                                <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                                  {new Date(trade.timestamp).toLocaleString()}
                                </td>
                                <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                                  {trade.type}
                                </td>
                                <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                                  ${trade.entry_price.toFixed(2)}
                                </td>
                                <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                                  ${trade.exit_price.toFixed(2)}
                                </td>
                                <td style={{ padding: '8px', borderBottom: '1px solid rgba(224, 224, 224, 1)' }}>
                                  {trade.size.toFixed(4)}
                                </td>
                                <td
                                  style={{
                                    padding: '8px',
                                    borderBottom: '1px solid rgba(224, 224, 224, 1)',
                                    color: trade.pnl >= 0 ? 'green' : 'red',
                                  }}
                                >
                                  ${trade.pnl.toFixed(2)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </Box>
                    </>
                  ) : (
                    <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 5 }}>
                      No trade data available.
                    </Typography>
                  )}
                </Box>
              )}

              {/* Monthly Returns Tab */}
              {tabValue === 3 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Monthly Returns
                  </Typography>
                  {monthlyReturnsData ? (
                    <Box sx={{ height: 400 }}>
                      <Bar options={barChartOptions} data={monthlyReturnsData} />
                    </Box>
                  ) : (
                    <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 5 }}>
                      No monthly returns data available.
                    </Typography>
                  )}
                </Box>
              )}
            </Paper>
          ) : (
            <Paper sx={{ p: 5, textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>
                No Backtest Results
              </Typography>
              <Typography variant="body1" color="textSecondary">
                Configure your backtest parameters and click "Run Backtest" to see results.
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Backtest;
