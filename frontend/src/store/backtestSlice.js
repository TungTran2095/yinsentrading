import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { toast } from 'react-toastify';
import api from '../services/api';

// Async thunks
export const runBacktest = createAsyncThunk(
  'backtest/runBacktest',
  async (backtestConfig, { rejectWithValue }) => {
    try {
      const response = await api.post('/backtest', backtestConfig);
      toast.success('Backtest completed successfully');
      return response.data;
    } catch (error) {
      toast.error(`Failed to run backtest: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

// Slice
const backtestSlice = createSlice({
  name: 'backtest',
  initialState: {
    results: null,
    loading: false,
    error: null,
  },
  reducers: {
    clearBacktestResults: (state) => {
      state.results = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // runBacktest
      .addCase(runBacktest.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(runBacktest.fulfilled, (state, action) => {
        state.results = action.payload;
        state.loading = false;
      })
      .addCase(runBacktest.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to run backtest' };
      });
  },
});

export const { clearBacktestResults } = backtestSlice.actions;

export default backtestSlice.reducer;
