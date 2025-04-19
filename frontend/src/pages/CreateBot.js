import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Button,
  Grid,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  FormHelperText,
  CircularProgress,
  Divider,
  Alert
} from '@mui/material';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { createBot } from '../store/botsSlice';
import { fetchModels } from '../store/modelsSlice';

// Validation schemas for each step
const strategySchema = Yup.object().shape({
  name: Yup.string().required('Bot name is required'),
  strategyType: Yup.string().required('Strategy type is required'),
  symbol: Yup.string().required('Trading pair is required'),
  timeframe: Yup.string().required('Timeframe is required'),
  ensembleId: Yup.string().when('strategyType', {
    is: (val) => val === 'ensemble' || val === 'combined',
    then: Yup.string().required('Ensemble model is required'),
  }),
  agentId: Yup.string().when('strategyType', {
    is: (val) => val === 'rl' || val === 'combined',
    then: Yup.string().required('RL agent is required'),
  }),
});

const executionSchema = Yup.object().shape({
  executionMode: Yup.string().required('Execution mode is required'),
  exchange: Yup.string().when('executionMode', {
    is: 'live',
    then: Yup.string().required('Exchange is required'),
  }),
  initialBalance: Yup.number().when('executionMode', {
    is: 'paper',
    then: Yup.number().required('Initial balance is required').positive('Must be positive'),
  }),
});

const riskSchema = Yup.object().shape({
  maxPositionSize: Yup.number().required('Max position size is required').min(0.01).max(1),
  maxDrawdown: Yup.number().required('Max drawdown is required').min(0.01).max(1),
  stopLossPct: Yup.number().required('Stop loss percentage is required').min(0.01).max(0.5),
  takeProfitPct: Yup.number().required('Take profit percentage is required').min(0.01).max(0.5),
});

const steps = ['Strategy Configuration', 'Execution Settings', 'Risk Management', 'Review'];

const CreateBot = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const { loading, error } = useSelector((state) => state.bots);
  const { ensembles, agents, loading: modelsLoading } = useSelector((state) => state.models);

  useEffect(() => {
    dispatch(fetchModels());
  }, [dispatch]);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleSubmit = (values) => {
    if (activeStep === steps.length - 1) {
      // Transform form values to API format
      const botConfig = {
        name: values.name,
        strategy: {
          type: values.strategyType,
          symbol: values.symbol,
          timeframe: values.timeframe,
          ensemble_id: values.ensembleId,
          agent_id: values.agentId,
          ensemble_weight: values.ensembleWeight,
          rl_weight: values.rlWeight,
          threshold_buy: values.thresholdBuy,
          threshold_sell: values.thresholdSell,
          confidence_threshold: values.confidenceThreshold,
          position_size: values.positionSize,
          stop_loss_pct: values.stopLossPct,
          take_profit_pct: values.takeProfitPct,
        },
        executor: {
          mode: values.executionMode,
          exchange: values.exchange,
          initial_balance: values.initialBalance,
          transaction_fee: values.transactionFee,
          slippage: values.slippage,
        },
        risk: {
          max_position_size: values.maxPositionSize,
          max_drawdown: values.maxDrawdown,
          stop_loss_pct: values.stopLossPct,
          take_profit_pct: values.takeProfitPct,
        },
      };

      dispatch(createBot(botConfig))
        .unwrap()
        .then((result) => {
          navigate(`/bots/${result.id}`);
        });
    } else {
      handleNext();
    }
  };

  const initialValues = {
    name: '',
    strategyType: 'combined',
    symbol: 'BTC/USDT',
    timeframe: '1h',
    ensembleId: '',
    agentId: '',
    ensembleWeight: 0.5,
    rlWeight: 0.5,
    thresholdBuy: 0.6,
    thresholdSell: 0.4,
    confidenceThreshold: 0.6,
    positionSize: 0.1,
    executionMode: 'paper',
    exchange: 'binance',
    initialBalance: 10000,
    transactionFee: 0.001,
    slippage: 0.001,
    maxPositionSize: 0.5,
    maxDrawdown: 0.2,
    stopLossPct: 0.05,
    takeProfitPct: 0.1,
  };

  const getStepContent = (step, formikProps) => {
    const { values, errors, touched, handleChange, setFieldValue } = formikProps;

    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                id="name"
                name="name"
                label="Bot Name"
                value={values.name}
                onChange={handleChange}
                error={touched.name && Boolean(errors.name)}
                helperText={touched.name && errors.name}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth error={touched.strategyType && Boolean(errors.strategyType)}>
                <InputLabel id="strategyType-label">Strategy Type</InputLabel>
                <Select
                  labelId="strategyType-label"
                  id="strategyType"
                  name="strategyType"
                  value={values.strategyType}
                  onChange={handleChange}
                  label="Strategy Type"
                >
                  <MenuItem value="ensemble">Ensemble Learning</MenuItem>
                  <MenuItem value="rl">Reinforcement Learning</MenuItem>
                  <MenuItem value="combined">Combined (Ensemble + RL)</MenuItem>
                </Select>
                {touched.strategyType && errors.strategyType && (
                  <FormHelperText>{errors.strategyType}</FormHelperText>
                )}
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="symbol"
                name="symbol"
                label="Trading Pair"
                value={values.symbol}
                onChange={handleChange}
                error={touched.symbol && Boolean(errors.symbol)}
                helperText={touched.symbol && errors.symbol}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth error={touched.timeframe && Boolean(errors.timeframe)}>
                <InputLabel id="timeframe-label">Timeframe</InputLabel>
                <Select
                  labelId="timeframe-label"
                  id="timeframe"
                  name="timeframe"
                  value={values.timeframe}
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
                {touched.timeframe && errors.timeframe && (
                  <FormHelperText>{errors.timeframe}</FormHelperText>
                )}
              </FormControl>
            </Grid>

            {(values.strategyType === 'ensemble' || values.strategyType === 'combined') && (
              <>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth error={touched.ensembleId && Boolean(errors.ensembleId)}>
                    <InputLabel id="ensembleId-label">Ensemble Model</InputLabel>
                    <Select
                      labelId="ensembleId-label"
                      id="ensembleId"
                      name="ensembleId"
                      value={values.ensembleId}
                      onChange={handleChange}
                      label="Ensemble Model"
                      disabled={modelsLoading}
                    >
                      {ensembles.map((model) => (
                        <MenuItem key={model.id} value={model.id}>
                          {model.name}
                        </MenuItem>
                      ))}
                    </Select>
                    {touched.ensembleId && errors.ensembleId && (
                      <FormHelperText>{errors.ensembleId}</FormHelperText>
                    )}
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    id="thresholdBuy"
                    name="thresholdBuy"
                    label="Buy Threshold"
                    type="number"
                    value={values.thresholdBuy}
                    onChange={handleChange}
                    inputProps={{ step: 0.05, min: 0, max: 1 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    id="thresholdSell"
                    name="thresholdSell"
                    label="Sell Threshold"
                    type="number"
                    value={values.thresholdSell}
                    onChange={handleChange}
                    inputProps={{ step: 0.05, min: 0, max: 1 }}
                  />
                </Grid>
              </>
            )}

            {(values.strategyType === 'rl' || values.strategyType === 'combined') && (
              <>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth error={touched.agentId && Boolean(errors.agentId)}>
                    <InputLabel id="agentId-label">RL Agent</InputLabel>
                    <Select
                      labelId="agentId-label"
                      id="agentId"
                      name="agentId"
                      value={values.agentId}
                      onChange={handleChange}
                      label="RL Agent"
                      disabled={modelsLoading}
                    >
                      {agents.map((agent) => (
                        <MenuItem key={agent.id} value={agent.id}>
                          {agent.name}
                        </MenuItem>
                      ))}
                    </Select>
                    {touched.agentId && errors.agentId && (
                      <FormHelperText>{errors.agentId}</FormHelperText>
                    )}
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    id="confidenceThreshold"
                    name="confidenceThreshold"
                    label="Confidence Threshold"
                    type="number"
                    value={values.confidenceThreshold}
                    onChange={handleChange}
                    inputProps={{ step: 0.05, min: 0, max: 1 }}
                  />
                </Grid>
              </>
            )}

            {values.strategyType === 'combined' && (
              <>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    id="ensembleWeight"
                    name="ensembleWeight"
                    label="Ensemble Weight"
                    type="number"
                    value={values.ensembleWeight}
                    onChange={(e) => {
                      const newValue = parseFloat(e.target.value);
                      setFieldValue('ensembleWeight', newValue);
                      setFieldValue('rlWeight', 1 - newValue);
                    }}
                    inputProps={{ step: 0.05, min: 0, max: 1 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    id="rlWeight"
                    name="rlWeight"
                    label="RL Weight"
                    type="number"
                    value={values.rlWeight}
                    onChange={(e) => {
                      const newValue = parseFloat(e.target.value);
                      setFieldValue('rlWeight', newValue);
                      setFieldValue('ensembleWeight', 1 - newValue);
                    }}
                    inputProps={{ step: 0.05, min: 0, max: 1 }}
                  />
                </Grid>
              </>
            )}

            <Grid item xs={12}>
              <TextField
                fullWidth
                id="positionSize"
                name="positionSize"
                label="Position Size (fraction of capital)"
                type="number"
                value={values.positionSize}
                onChange={handleChange}
                inputProps={{ step: 0.05, min: 0.01, max: 1 }}
              />
            </Grid>
          </Grid>
        );
      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControl fullWidth error={touched.executionMode && Boolean(errors.executionMode)}>
                <InputLabel id="executionMode-label">Execution Mode</InputLabel>
                <Select
                  labelId="executionMode-label"
                  id="executionMode"
                  name="executionMode"
                  value={values.executionMode}
                  onChange={handleChange}
                  label="Execution Mode"
                >
                  <MenuItem value="paper">Paper Trading (Simulation)</MenuItem>
                  <MenuItem value="live">Live Trading</MenuItem>
                </Select>
                {touched.executionMode && errors.executionMode && (
                  <FormHelperText>{errors.executionMode}</FormHelperText>
                )}
              </FormControl>
            </Grid>

            {values.executionMode === 'live' && (
              <Grid item xs={12}>
                <FormControl fullWidth error={touched.exchange && Boolean(errors.exchange)}>
                  <InputLabel id="exchange-label">Exchange</InputLabel>
                  <Select
                    labelId="exchange-label"
                    id="exchange"
                    name="exchange"
                    value={values.exchange}
                    onChange={handleChange}
                    label="Exchange"
                  >
                    <MenuItem value="binance">Binance</MenuItem>
                    <MenuItem value="ftx">FTX</MenuItem>
                    <MenuItem value="bybit">Bybit</MenuItem>
                  </Select>
                  {touched.exchange && errors.exchange && (
                    <FormHelperText>{errors.exchange}</FormHelperText>
                  )}
                </FormControl>
              </Grid>
            )}

            {values.executionMode === 'paper' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  id="initialBalance"
                  name="initialBalance"
                  label="Initial Balance (USD)"
                  type="number"
                  value={values.initialBalance}
                  onChange={handleChange}
                  error={touched.initialBalance && Boolean(errors.initialBalance)}
                  helperText={touched.initialBalance && errors.initialBalance}
                  inputProps={{ step: 1000, min: 100 }}
                />
              </Grid>
            )}

            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="transactionFee"
                name="transactionFee"
                label="Transaction Fee"
                type="number"
                value={values.transactionFee}
                onChange={handleChange}
                inputProps={{ step: 0.0001, min: 0, max: 0.01 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="slippage"
                name="slippage"
                label="Slippage"
                type="number"
                value={values.slippage}
                onChange={handleChange}
                inputProps={{ step: 0.0001, min: 0, max: 0.01 }}
              />
            </Grid>
          </Grid>
        );
      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="maxPositionSize"
                name="maxPositionSize"
                label="Max Position Size (fraction of capital)"
                type="number"
                value={values.maxPositionSize}
                onChange={handleChange}
                error={touched.maxPositionSize && Boolean(errors.maxPositionSize)}
                helperText={touched.maxPositionSize && errors.maxPositionSize}
                inputProps={{ step: 0.05, min: 0.01, max: 1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="maxDrawdown"
                name="maxDrawdown"
                label="Max Drawdown"
                type="number"
                value={values.maxDrawdown}
                onChange={handleChange}
                error={touched.maxDrawdown && Boolean(errors.maxDrawdown)}
                helperText={touched.maxDrawdown && errors.maxDrawdown}
                inputProps={{ step: 0.05, min: 0.01, max: 1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="stopLossPct"
                name="stopLossPct"
                label="Stop Loss Percentage"
                type="number"
                value={values.stopLossPct}
                onChange={handleChange}
                error={touched.stopLossPct && Boolean(errors.stopLossPct)}
                helperText={touched.stopLossPct && errors.stopLossPct}
                inputProps={{ step: 0.01, min: 0.01, max: 0.5 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                id="takeProfitPct"
                name="takeProfitPct"
                label="Take Profit Percentage"
                type="number"
                value={values.takeProfitPct}
                onChange={handleChange}
                error={touched.takeProfitPct && Boolean(errors.takeProfitPct)}
                helperText={touched.takeProfitPct && errors.takeProfitPct}
                inputProps={{ step: 0.01, min: 0.01, max: 0.5 }}
              />
            </Grid>
          </Grid>
        );
      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Bot Configuration Summary
            </Typography>
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Strategy Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Bot Name:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.name}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Strategy Type:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.strategyType}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Trading Pair:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.symbol}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Timeframe:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.timeframe}</Typography>
                </Grid>
              </Grid>
            </Paper>

            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Execution Settings
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Execution Mode:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.executionMode}</Typography>
                </Grid>
                {values.executionMode === 'live' && (
                  <>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Exchange:
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">{values.exchange}</Typography>
                    </Grid>
                  </>
                )}
                {values.executionMode === 'paper' && (
                  <>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Initial Balance:
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">${values.initialBalance}</Typography>
                    </Grid>
                  </>
                )}
              </Grid>
            </Paper>

            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Risk Management
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Max Position Size:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.maxPositionSize * 100}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Max Drawdown:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.maxDrawdown * 100}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Stop Loss:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.stopLossPct * 100}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Take Profit:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{values.takeProfitPct * 100}%</Typography>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  const getValidationSchema = (step) => {
    switch (step) {
      case 0:
        return strategySchema;
      case 1:
        return executionSchema;
      case 2:
        return riskSchema;
      default:
        return null;
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom component="div">
        Create Trading Bot
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error.message}
        </Alert>
      )}

      <Paper sx={{ p: 3 }}>
        <Formik
          initialValues={initialValues}
          validationSchema={getValidationSchema(activeStep)}
          onSubmit={handleSubmit}
        >
          {(formikProps) => (
            <Form>
              {getStepContent(activeStep, formikProps)}
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
                {activeStep !== 0 && (
                  <Button onClick={handleBack} sx={{ mr: 1 }}>
                    Back
                  </Button>
                )}
                <Button
                  variant="contained"
                  color="primary"
                  type={activeStep === steps.length - 1 ? 'submit' : 'button'}
                  onClick={activeStep === steps.length - 1 ? undefined : () => formikProps.submitForm()}
                  disabled={loading}
                >
                  {activeStep === steps.length - 1 ? 'Create Bot' : 'Next'}
                  {loading && activeStep === steps.length - 1 && (
                    <CircularProgress size={24} sx={{ ml: 1 }} />
                  )}
                </Button>
              </Box>
            </Form>
          )}
        </Formik>
      </Paper>
    </Box>
  );
};

export default CreateBot;
